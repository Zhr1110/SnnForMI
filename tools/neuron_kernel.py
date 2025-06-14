import logging

try:
    import cupy
except BaseException as e:
    logging.info(f'spikingjelly.activation_based.neuron_kernel: {e}')
    cupy = None

import torch
import torch.nn.functional as F
from .auto_cuda import cuda_utils, configure
from . import  surrogate, tensor_cache
import numpy as np


class MultiStepIFNodePTT(torch.autograd.Function):
    @staticmethod
    def create_fptt_kernel(hard_reset: bool, dtype: str):
        kernel_name = f'IFNode_fptt_{"hard" if hard_reset else "soft"}Reset_{dtype}'

        if dtype == 'fp32':
            code = rf'''
            extern "C" __global__
            void {kernel_name}(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq, 
            const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel) 
            '''

            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < neuron_num)
            {
                const int dt = neuron_num;
                for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                {
                    const int t = index + mem_offset;
                    h_seq[t] = v_v_seq[t] + x_seq[t];
                    if (h_seq[t] >= v_threshold)
            '''

            if hard_reset:
                code += r'''
                    {
                        spike_seq[t] = 1.0f;
                        v_v_seq[t + dt] = v_reset;
                    }
                '''
            else:
                code += r'''
                    {
                        spike_seq[t] = 1.0f;
                        v_v_seq[t + dt] = h_seq[t] - v_threshold;
                    }
                '''

            code += r'''
                    else
                    {
                        spike_seq[t] = 0.0f;
                        v_v_seq[t + dt] = h_seq[t];
                    }
                }
            }
            }
            '''

        elif dtype == 'fp16':
            code = rf'''
            #include <cuda_fp16.h>
            extern "C" __global__
            void {kernel_name}(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq, 
            const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel) 
            '''

            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = neuron_num >> 1;
            if (index < stride)
            {
                const int numel_2 = numel >> 1;
                const half2 v_threshold_half2 = __half2half2(v_threshold);
            '''

            if hard_reset:
                code += r'''
                    const half2 v_reset_half2 = __half2half2(v_reset);
                '''

            code += r'''
                for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                {
                    const int t = index + mem_offset;
                    h_seq[t] = __hadd2(v_v_seq[t], x_seq[t]);

                    spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
            '''

            if hard_reset:
                code += r'''
                    v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                '''
            else:
                code += r'''
                    v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                '''

            code += r'''
                }
            }
            }
            '''
        else:
            raise TypeError

        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options,
                              backend=configure.cuda_compiler_backend)

    @staticmethod
    def create_bptt_kernel(sg_cuda_code_fun, hard_reset: bool, detach_reset: bool, dtype: str):

        kernel_name = f'IFNode_bptt_{"hard" if hard_reset else "soft"}Reset_{"detachReset" if detach_reset else ""}_{dtype}'

        code_grad_s_to_h = sg_cuda_code_fun(x='over_th', y='grad_s_to_h', dtype=dtype)

        if dtype == 'fp32':
            code = fr'''
            extern "C" __global__
            void {kernel_name}(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
            const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''

            code += r'''
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {   
                    float grad_h = 0.0f;  // grad_h will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const float over_th = h_seq[t] - v_threshold;
            '''
            code += code_grad_s_to_h
            if detach_reset:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike_seq[t];
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f;
                    '''
            else:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(grad_s_to_h, v_reset - h_seq[t], 1.0f - spike_seq[t]);
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                    '''

            code += code_grad_v_to_h
            code += r'''
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, (grad_v_seq[t] + grad_h) * grad_v_to_h);
                grad_x_seq[t] = grad_h;
                }
            grad_v_init[index] = grad_h;
            }
            }
            '''

        elif dtype == 'fp16':
            code = fr'''
            #include <cuda_fp16.h>
            extern "C" __global__
            void {kernel_name}(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
            const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''
            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = neuron_num >> 1;
            if (index < stride)
            {   
                const half2 v_threshold_half2 = __half2half2(v_threshold);
            '''

            if hard_reset:
                code += r'''
                    const half2 v_reset_half2 = __half2half2(v_reset);
                '''

            code += r'''
                half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                {
                    const int t = index + mem_offset;
                    const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
            '''
            code += code_grad_s_to_h

            if detach_reset:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __float2half2_rn(1.0f);
                    '''
            else:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]), grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                    '''

            code += code_grad_v_to_h
            code += r'''
                    grad_h = __hfma2(__hadd2(grad_v_seq[t], grad_h), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
                    grad_x_seq[t] = grad_h;
                    }
            grad_v_init[index] = grad_h;
            }
            }
            '''
        else:
            raise TypeError
        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options,
                              backend=configure.cuda_compiler_backend)

    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_threshold: float, v_reset: float,
                detach_reset: bool, sg_cuda_code_fun):
        requires_grad = x_seq.requires_grad or v_init.requires_grad
        device = x_seq.get_device()
        if x_seq.dtype == torch.float32:
            dtype = 'fp32'
            cp_dtype = np.float32
        elif x_seq.dtype == torch.float16:
            dtype = 'fp16'
            cp_dtype = np.half
        else:
            raise NotImplementedError

        use_pad = False
        if dtype == 'fp16' and v_init.numel() % 2 != 0:
            # only fp16 needs even numel because we use half2 to accelerate
            # when numel is odd, we will pad x_seq
            use_pad = True
            x_seq = F.pad(x_seq, (0, 1))  # [T, N] -> [T, N + 1]
            v_init = F.pad(v_init, (0, 1))  # [N] -> [N + 1]

        zero_shape = list(x_seq.shape)
        zero_shape[0] *= 3
        v_seq, h_seq, spike_seq = torch.split(torch.zeros(zero_shape, device=x_seq.device, dtype=x_seq.dtype),
                                              x_seq.shape[0])

        v_v_seq = torch.cat((v_init.unsqueeze(0), v_seq))

        with cuda_utils.DeviceEnvironment(device):
            numel = x_seq.numel()
            neuron_num = numel // x_seq.shape[0]

            threads = configure.cuda_threads
            if dtype == 'fp16':
                assert neuron_num % 2 == 0
                blocks = cuda_utils.cal_blocks(neuron_num >> 1)
                # we will take two neurons to calculate as one neuron in cuda half2
            else:
                blocks = cuda_utils.cal_blocks(neuron_num)
            cp_numel = cupy.asarray(numel)
            cp_neuron_num = cupy.asarray(neuron_num)
            cp_v_threshold = cupy.asarray(v_threshold, dtype=cp_dtype)
            if v_reset is None:
                cp_v_reset = None
                hard_reset = False
                x_seq, v_v_seq, h_seq, spike_seq, cp_v_threshold, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(
                    x_seq, v_v_seq, h_seq, spike_seq, cp_v_threshold, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_v_threshold, cp_neuron_num, cp_numel]
            else:
                cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                hard_reset = True
                x_seq, v_v_seq, h_seq, spike_seq, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(
                    x_seq, v_v_seq, h_seq, spike_seq, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_v_threshold, cp_v_reset, cp_neuron_num,
                               cp_numel]

            kernel = MultiStepIFNodePTT.create_fptt_kernel(hard_reset, dtype)

            kernel(
                (blocks,), (threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )

        if requires_grad:
            ctx.use_pad = use_pad
            if configure.save_spike_as_bool_in_neuron_kernel:
                ctx.s_shape = spike_seq.shape
                ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(spike_seq)
                ctx.save_for_backward(h_seq)
            else:
                ctx.save_for_backward(h_seq, spike_seq)
            ctx.blocks = blocks
            ctx.threads = threads
            ctx.cp_numel = cp_numel
            ctx.cp_neuron_num = cp_neuron_num
            ctx.cp_v_threshold = cp_v_threshold
            ctx.cp_v_reset = cp_v_reset
            ctx.detach_reset = detach_reset
            ctx.sg_cuda_code_fun = sg_cuda_code_fun

        if use_pad:
            return spike_seq[..., :-1], v_v_seq[1:, ..., :-1]
        else:
            return spike_seq, v_v_seq[1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        if ctx.use_pad:
            # grad_spike_seq.shape = [T, N]
            # grad_v_seq.shape = [T, N]
            # h_seq.shape = [T, N + 1]
            # spike_seq.shape = [T, N + 1]
            grad_spike_seq = F.pad(grad_spike_seq, (0, 1))
            grad_v_seq = F.pad(grad_v_seq, (0, 1))

        device = grad_spike_seq.get_device()

        if configure.save_spike_as_bool_in_neuron_kernel:
            h_seq = ctx.saved_tensors[0]
            spike_seq = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
        else:
            h_seq, spike_seq = ctx.saved_tensors

        zero_shape = list(grad_spike_seq.shape)
        zero_shape[0] += 1
        zero_data = torch.zeros(zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype)
        grad_x_seq = zero_data[0: -1]
        grad_v_init = zero_data[-1]

        if ctx.cp_v_reset is None:
            hard_reset = False
        else:
            hard_reset = True

        if grad_spike_seq.dtype == torch.float32:
            dtype = 'fp32'
        elif grad_spike_seq.dtype == torch.float16:
            dtype = 'fp16'
        else:
            raise NotImplementedError

        kernel = MultiStepIFNodePTT.create_bptt_kernel(ctx.sg_cuda_code_fun, hard_reset, ctx.detach_reset, dtype)

        with cuda_utils.DeviceEnvironment(device):

            if hard_reset:
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_init, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_init, ctx.cp_v_threshold,
                    ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_init,
                               ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel]
            else:
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_init, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_init, ctx.cp_v_threshold,
                    ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_init,
                               ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel]

            kernel(
                (ctx.blocks,), (ctx.threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )
        if ctx.use_pad:
            return grad_x_seq[..., :-1], grad_v_init[..., :-1], None, None, None, None
        else:
            return grad_x_seq, grad_v_init, None, None, None, None


class MultiStepLIFNodePTT(torch.autograd.Function):
    @staticmethod
    def create_fptt_kernel(decay_input: bool, hard_reset: bool, dtype: str, kernel_name_prefix: str = 'LIFNode'):
        kernel_name = f'{kernel_name_prefix}_fptt_decayInput{decay_input}_{"hard" if hard_reset else "soft"}Reset_{dtype}'

        if dtype == 'fp32':
            code = rf'''
            extern "C" __global__
            void {kernel_name}(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
            const float & reciprocal_tau, 
            const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''
            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < neuron_num)
            {
                const int dt = neuron_num;
                for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                {
                    const int t = index + mem_offset;
            '''

            if hard_reset:
                if decay_input:
                    code += r'''
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_reset);
                    '''
                else:
                    code += r'''
                        h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                    '''
                code += r'''
                    if (h_seq[t] >= v_threshold)
                    {
                        spike_seq[t] = 1.0f;
                        v_v_seq[t + dt] = v_reset;
                    }
                '''
            else:
                if decay_input:
                    code += r'''
                        h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t]);
                    '''
                else:
                    code += r'''
                        h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                    '''
                code += r'''
                    if (h_seq[t] >= v_threshold)
                    {
                        spike_seq[t] = 1.0f;
                        v_v_seq[t + dt] = h_seq[t] - v_threshold;
                    }
                '''

            code += r'''
                    else
                    {
                        spike_seq[t] = 0.0f;
                        v_v_seq[t + dt] = h_seq[t];
                    }

                }
            }
            }
            '''

        elif dtype == 'fp16':
            code = rf'''
            #include <cuda_fp16.h>
            extern "C" __global__
            void {kernel_name}(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
            const half & reciprocal_tau, 
            const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel) 
            '''

            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = neuron_num >> 1;
            if (index < stride)
            {
                const int numel_2 = numel >> 1;
                const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                const half2 v_threshold_half2 = __half2half2(v_threshold);
            '''

            if hard_reset:
                code += r'''
                    const half2 v_reset_half2 = __half2half2(v_reset);
                '''

            code += r'''
                for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                {
                    const int t = index + mem_offset;
            '''
            if hard_reset:
                if decay_input:
                    code += r'''
                        h_seq[t] = __hfma2(__hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_reset_half2), reciprocal_tau_half2, v_v_seq[t]);
                    '''
                else:
                    code += r'''
                        // h_seq[t] = v_v_seq[t] - reciprocal_tau * (v_v_seq[t] - v_reset) + x_seq[t];
                        // = reciprocal_tau * (v_reset - v_v_seq[t]) + v_v_seq[t] + x_seq[t];
                        h_seq[t] = __hadd2(__hfma2(__hsub2(v_reset_half2,  v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]), x_seq[t]);
                    '''
                code += r'''
                    spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                    v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                '''
            else:
                if decay_input:
                    code += r'''
                        h_seq[t] = __hfma2(__hsub2(x_seq[t], v_v_seq[t]), reciprocal_tau_half2, v_v_seq[t]);
                    '''
                else:
                    code += r'''
                        // h_seq[t] = v_v_seq[t] * (1.0f - reciprocal_tau) + x_seq[t];
                        h_seq[t] = __hfma2(__hsub2(__float2half2_rn(1.0f), reciprocal_tau_half2), v_v_seq[t], x_seq[t]);
                    '''
                code += r'''
                    spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
                    v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                '''

            code += r'''
                }
            }
            }
            '''
        else:
            raise TypeError
        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options,
                              backend=configure.cuda_compiler_backend)

    @staticmethod
    def create_bptt_kernel(sg_cuda_code_fun, decay_input: bool, hard_reset: bool, detach_reset: bool, dtype: str):

        kernel_name = f'LIFNode_bptt_decayInput{decay_input}_{"hard" if hard_reset else "soft"}Reset_{"detachReset" if detach_reset else ""}_{dtype}'

        code_grad_s_to_h = sg_cuda_code_fun(x='over_th', y='grad_s_to_h', dtype=dtype)

        if dtype == 'fp32':
            code = fr'''
            extern "C" __global__
            void {kernel_name}(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq,
            float* grad_x_seq, float* grad_v_init,
            const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
            const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''

            code += r'''
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {   
                    float grad_h = 0.0f;  // grad_h will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const float over_th = h_seq[t] - v_threshold;
            '''
            code += code_grad_s_to_h
            if detach_reset:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike_seq[t];
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f;
                    '''
            else:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(-grad_s_to_h, v_threshold, 1.0f);
                    '''

            code += code_grad_v_to_h
            code += r'''
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
            '''
            if decay_input:
                code += r'''
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                '''
            else:
                code += r'''
                    grad_x_seq[t] = grad_h;
                '''
            code += r'''
                }
            grad_v_init[index] = grad_h * one_sub_reciprocal_tau;
            }
            }
            '''

        elif dtype == 'fp16':
            code = fr'''
            #include <cuda_fp16.h>
            extern "C" __global__
            void {kernel_name}(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq,
            half2* grad_x_seq, half2* grad_v_init,
            const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
            const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''
            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = neuron_num >> 1;
            if (index < stride)
            {   
                const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                const half2 v_threshold_half2 = __half2half2(v_threshold);
            '''

            if hard_reset:
                code += r'''
                    const half2 v_reset_half2 = __half2half2(v_reset);
                '''

            code += r'''
                half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                {
                    const int t = index + mem_offset;

                    const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
            '''

            code += code_grad_s_to_h

            if detach_reset:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __float2half2_rn(1.0f);
                    '''
            else:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                    '''

            code += code_grad_v_to_h
            code += r'''                        
                    grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
            '''
            if decay_input:
                code += r''' 
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                '''
            else:
                code += r''' 
                        grad_x_seq[t] = grad_h;
                '''
            code += r'''
                }
            grad_v_init[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
            }
            }
            '''

        else:
            raise TypeError
        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options,
                              backend=configure.cuda_compiler_backend)

    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, decay_input: bool, tau: float, v_threshold: float,
                v_reset: float,
                detach_reset: bool, sg_cuda_code_fun):
        requires_grad = x_seq.requires_grad or v_init.requires_grad
        device = x_seq.get_device()
        if x_seq.dtype == torch.float32:
            dtype = 'fp32'
            cp_dtype = np.float32
        elif x_seq.dtype == torch.float16:
            dtype = 'fp16'
            cp_dtype = np.half
        else:
            raise NotImplementedError

        use_pad = False
        if dtype == 'fp16' and v_init.numel() % 2 != 0:
            # only fp16 needs even numel because we use half2 to accelerate
            # when numel is odd, we will pad x_seq
            use_pad = True
            x_seq = F.pad(x_seq, (0, 1))  # [T, N] -> [T, N + 1]
            v_init = F.pad(v_init, (0, 1))  # [N] -> [N + 1]

        zero_shape = list(x_seq.shape)
        zero_shape[0] *= 3
        v_seq, h_seq, spike_seq = torch.split(torch.zeros(zero_shape, device=x_seq.device, dtype=x_seq.dtype),
                                              x_seq.shape[0])

        v_v_seq = torch.cat((v_init.unsqueeze(0), v_seq))

        with cuda_utils.DeviceEnvironment(device):
            numel = x_seq.numel()
            neuron_num = numel // x_seq.shape[0]

            threads = configure.cuda_threads
            if dtype == 'fp16':
                assert neuron_num % 2 == 0
                blocks = cuda_utils.cal_blocks(neuron_num >> 1)
                # we will take two neurons to calculate as one neuron in cuda half2
            else:
                blocks = cuda_utils.cal_blocks(neuron_num)

            cp_numel = cupy.asarray(numel)
            cp_neuron_num = cupy.asarray(neuron_num)
            cp_v_threshold = cupy.asarray(v_threshold, dtype=cp_dtype)
            cp_reciprocal_tau = cupy.asarray(1. / tau, dtype=cp_dtype)
            cp_one_sub_reciprocal_tau = cupy.asarray(1. - 1. / tau, dtype=cp_dtype)

            if v_reset is None:
                cp_v_reset = None
                hard_reset = False
                x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(
                    x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num,
                               cp_numel]
            else:
                cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                hard_reset = True
                x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(
                    x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset, cp_neuron_num,
                    cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset,
                               cp_neuron_num, cp_numel]

            kernel = MultiStepLIFNodePTT.create_fptt_kernel(decay_input, hard_reset, dtype)
            kernel(
                (blocks,), (threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )

        if requires_grad:
            ctx.decay_input = decay_input
            ctx.use_pad = use_pad
            if configure.save_spike_as_bool_in_neuron_kernel:
                ctx.s_shape = spike_seq.shape
                ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(spike_seq)
                ctx.save_for_backward(h_seq)
            else:
                ctx.save_for_backward(h_seq, spike_seq)
            ctx.blocks = blocks
            ctx.threads = threads
            ctx.cp_numel = cp_numel
            ctx.cp_neuron_num = cp_neuron_num
            ctx.cp_reciprocal_tau = cp_reciprocal_tau
            ctx.cp_one_sub_reciprocal_tau = cp_one_sub_reciprocal_tau
            ctx.cp_v_threshold = cp_v_threshold
            ctx.cp_v_reset = cp_v_reset
            ctx.detach_reset = detach_reset
            ctx.sg_cuda_code_fun = sg_cuda_code_fun

        if use_pad:
            return spike_seq[..., :-1], v_v_seq[1:, ..., :-1]
        else:
            return spike_seq, v_v_seq[1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        if ctx.use_pad:
            # grad_spike_seq.shape = [T, N]
            # grad_v_seq.shape = [T, N]
            # h_seq.shape = [T, N + 1]
            # spike_seq.shape = [T, N + 1]
            grad_spike_seq = F.pad(grad_spike_seq, (0, 1))
            grad_v_seq = F.pad(grad_v_seq, (0, 1))

        device = grad_spike_seq.get_device()
        if configure.save_spike_as_bool_in_neuron_kernel:
            h_seq = ctx.saved_tensors[0]
            spike_seq = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
        else:
            h_seq, spike_seq = ctx.saved_tensors
        zero_shape = list(grad_spike_seq.shape)
        zero_shape[0] += 1
        zero_data = torch.zeros(zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype)
        grad_x_seq = zero_data[0: -1]
        grad_v_init = zero_data[-1]

        if ctx.cp_v_reset is None:
            hard_reset = False
        else:
            hard_reset = True

        if grad_spike_seq.dtype == torch.float32:
            dtype = 'fp32'
        elif grad_spike_seq.dtype == torch.float16:
            dtype = 'fp16'
        else:
            raise NotImplementedError

        kernel = MultiStepLIFNodePTT.create_bptt_kernel(ctx.sg_cuda_code_fun, ctx.decay_input, hard_reset,
                                                        ctx.detach_reset, dtype)

        with cuda_utils.DeviceEnvironment(device):

            if hard_reset:
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_init, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_init, ctx.cp_reciprocal_tau,
                    ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num,
                    ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_init,
                               ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold,
                               ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel]
            else:
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_init, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_init, ctx.cp_reciprocal_tau,
                    ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, grad_x_seq, grad_v_init,
                               ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold,
                               ctx.cp_neuron_num, ctx.cp_numel]

            kernel(
                (ctx.blocks,), (ctx.threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )
        if ctx.use_pad:
            return grad_x_seq[..., :-1], grad_v_init[..., :-1], None, None, None, None, None, None
        else:
            return grad_x_seq, grad_v_init, None, None, None, None, None, None


class MultiStepParametricLIFNodePTT(torch.autograd.Function):
    @staticmethod
    def create_fptt_kernel(decay_input: bool, hard_reset: bool, dtype: str):
        return MultiStepLIFNodePTT.create_fptt_kernel(decay_input, hard_reset, dtype,
                                                      kernel_name_prefix='ParametricLIFNode')

    @staticmethod
    def create_bptt_kernel(sg_cuda_code_fun, decay_input: bool, hard_reset: bool, detach_reset: bool, dtype: str):
        kernel_name = f'ParametricLIFNode_bptt_decayInput{decay_input}_{"hard" if hard_reset else "soft"}Reset_{"detachReset" if detach_reset else ""}_{dtype}'

        code_grad_s_to_h = sg_cuda_code_fun(x='over_th', y='grad_s_to_h', dtype=dtype)

        if dtype == 'fp32':
            code = fr'''
            extern "C" __global__
            void {kernel_name}(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
            float* grad_x_seq, float* grad_v_init, float* grad_reciprocal_tau,
            const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
            const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''
            code += r'''
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
            '''
            code += f'__shared__ float sdata[{configure.cuda_threads}];'
            code += r'''
                if (index < neuron_num)
                {   
                    float grad_h = 0.0f;  // grad_h will be used recursively
                    sdata[threadIdx.x] = 0.0f;
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const float over_th = h_seq[t] - v_threshold;
            '''
            code += code_grad_s_to_h
            if detach_reset:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike_seq[t];
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f;
                    '''
            else:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(v_reset - h_seq[t], grad_s_to_h, 1.0f - spike_seq[t]);
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                    // const float grad_v_to_h = fmaf(-v_threshold, grad_s_to_h, 1.0f);
                    '''

            code += code_grad_v_to_h
            code += r'''
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * one_sub_reciprocal_tau) * grad_v_to_h;
                // grad_h = fmaf(grad_spike_seq[t], grad_s_to_h, fmaf(grad_h, one_sub_reciprocal_tau, grad_v_seq[t]) * grad_v_to_h);
            '''
            if decay_input:
                code += r'''
                    grad_x_seq[t] = grad_h * reciprocal_tau;
                    sdata[threadIdx.x] += grad_h * (h_seq[t] - v_v_seq[t]) / reciprocal_tau;
                '''
            else:
                if hard_reset:
                    code += r'''
                        grad_x_seq[t] = grad_h;
                        sdata[threadIdx.x] += grad_h * (v_reset - v_v_seq[t]);
                    '''
                else:
                    code += r'''
                        grad_x_seq[t] = grad_h;
                        sdata[threadIdx.x] -= grad_h * v_v_seq[t];
                    '''
            code += r'''
                }
            grad_v_init[index] = grad_h * one_sub_reciprocal_tau;
            }
            else
            {
                sdata[threadIdx.x] = 0.0f;
            }
            int threadx = blockDim.x;
            #pragma unroll
            for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
            {
            // Synchronize all thread before next loop
            __syncthreads();
            if (threadIdx.x < stride)
            {
                sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            }
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
            atomicAdd(grad_reciprocal_tau, sdata[0]);
            }
            }
            '''

        elif dtype == 'fp16':
            code = fr'''
            #include <cuda_fp16.h>
            extern "C" __global__
            void {kernel_name}(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
            half2* grad_x_seq, half2* grad_v_init,  float* grad_reciprocal_tau,
            const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
            const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)\
            // note that grad_reciprocal_tau is float to avoid overflow
            '''
            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = neuron_num >> 1;

            '''
            code += f'__shared__ float sdata[{configure.cuda_threads}];'
            code += r'''
            if (index < stride)
            {   
                const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                const half2 v_threshold_half2 = __half2half2(v_threshold);
            '''

            if hard_reset:
                code += r'''
                    const half2 v_reset_half2 = __half2half2(v_reset);
                '''

            code += r'''

                half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                sdata[threadIdx.x] = 0.0f;
                for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                {
                    const int t = index + mem_offset;

                    const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);

            '''
            code += code_grad_s_to_h

            if detach_reset:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __float2half2_rn(1.0f);
                    '''
            else:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                    '''

            code += code_grad_v_to_h
            code += r'''                        
                    grad_h = __hfma2(__hfma2(grad_h, one_sub_reciprocal_tau_half2, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));
            '''
            if decay_input:
                code += r'''  
                        grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                        half2 temp_sum = __h2div(__hmul2(grad_h, __hsub2(h_seq[t], v_v_seq[t])), reciprocal_tau_half2);
                        sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                '''
            else:
                if hard_reset:
                    code += r'''  
                            grad_x_seq[t] = grad_h;
                            half2 temp_sum = __hmul2(grad_h, __hsub2(v_reset_half2, v_v_seq[t]));
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                    '''
                else:
                    code += r'''  
                            grad_x_seq[t] = grad_h;
                            half2 temp_sum = __hmul2(grad_h, __hneg2(v_v_seq[t]));
                            sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_sum), __high2half(temp_sum)));
                    '''
            code += r'''  
                }
            grad_v_init[index] = __hmul2(grad_h, one_sub_reciprocal_tau_half2);
            }
            else
            {
                sdata[threadIdx.x] = 0.0f;
            }
            int threadx = blockDim.x;
            #pragma unroll
            for (int i = threadx >> 1; i > 0; i = i >> 1)
            {
            // Synchronize all thread before next loop
            __syncthreads();
            if (threadIdx.x < i)
            {
                sdata[threadIdx.x] += sdata[threadIdx.x + i];
            }
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {                
            /*
            The 32-bit floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher.

            The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher.

            The 32-bit __half2 floating-point version of atomicAdd() is only supported by devices of compute capability 6.x and higher. The atomicity of the __half2 or __nv_bfloat162 add operation is guaranteed separately for each of the two __half or __nv_bfloat16 elements; the entire __half2 or __nv_bfloat162 is not guaranteed to be atomic as a single 32-bit access.

            The 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher.

            The 16-bit __nv_bfloat16 floating-point version of atomicAdd() is only supported by devices of compute capability 8.x and higher.
            */

            atomicAdd(grad_reciprocal_tau, sdata[0]);

            }
            }
            '''
        else:
            raise TypeError

        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options,
                              backend=configure.cuda_compiler_backend)

    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, reciprocal_tau: torch.Tensor, decay_input: bool,
                v_threshold: float,
                v_reset: float, detach_reset: bool, sg_cuda_code_fun):
        # reciprocal_tau.dtype is float32 even when using amp
        requires_grad = x_seq.requires_grad or v_init.requires_grad
        device = x_seq.get_device()
        if x_seq.dtype == torch.float32:
            dtype = 'fp32'
            cp_dtype = np.float32
        elif x_seq.dtype == torch.float16:
            dtype = 'fp16'
            cp_dtype = np.half
            # assert torch.cuda.get_device_capability(device)[0] >= 7, "MultiStepParametricLIFNodePTT can not run in the current device with float16 because the 16-bit __half floating-point version of atomicAdd() is only supported by devices of compute capability 7.x and higher."

        else:
            raise NotImplementedError

        use_pad = False
        if dtype == 'fp16' and v_init.numel() % 2 != 0:
            # only fp16 needs even numel because we use half2 to accelerate
            # when numel is odd, we will pad x_seq
            use_pad = True
            x_seq = F.pad(x_seq, (0, 1))  # [T, N] -> [T, N + 1]
            v_init = F.pad(v_init, (0, 1))  # [N] -> [N + 1]

        zero_shape = list(x_seq.shape)
        zero_shape[0] *= 3
        v_seq, h_seq, spike_seq = torch.split(torch.zeros(zero_shape, device=x_seq.device, dtype=x_seq.dtype),
                                              x_seq.shape[0])

        v_v_seq = torch.cat((v_init.unsqueeze(0), v_seq))
        tau = 1. / reciprocal_tau.item()

        with cuda_utils.DeviceEnvironment(device):
            numel = x_seq.numel()
            neuron_num = numel // x_seq.shape[0]

            threads = configure.cuda_threads
            if dtype == 'fp16':
                assert neuron_num % 2 == 0
                blocks = cuda_utils.cal_blocks(neuron_num >> 1)
                # we will take two neurons to calculate as one neuron in cuda half2
            else:
                blocks = cuda_utils.cal_blocks(neuron_num)

            cp_numel = cupy.asarray(numel)
            cp_neuron_num = cupy.asarray(neuron_num)
            cp_v_threshold = cupy.asarray(v_threshold, dtype=cp_dtype)
            cp_reciprocal_tau = cupy.asarray(1. / tau, dtype=cp_dtype)
            cp_one_sub_reciprocal_tau = cupy.asarray(1. - 1. / tau, dtype=cp_dtype)

            if v_reset is None:
                cp_v_reset = None
                hard_reset = False
                x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(
                    x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_neuron_num,
                               cp_numel]
            else:
                cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                hard_reset = True
                x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(
                    x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset, cp_neuron_num,
                    cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_threshold, cp_v_reset,
                               cp_neuron_num, cp_numel]

            kernel = MultiStepParametricLIFNodePTT.create_fptt_kernel(decay_input, hard_reset, dtype)

            kernel(
                (blocks,), (threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )

        if requires_grad:
            ctx.decay_input = decay_input
            ctx.use_pad = use_pad
            if configure.save_spike_as_bool_in_neuron_kernel:
                ctx.s_shape = spike_seq.shape
                ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(spike_seq)
                ctx.save_for_backward(h_seq, v_v_seq)
            else:
                ctx.save_for_backward(h_seq, spike_seq, v_v_seq)
            ctx.blocks = blocks
            ctx.threads = threads
            ctx.cp_numel = cp_numel
            ctx.cp_neuron_num = cp_neuron_num
            ctx.cp_reciprocal_tau = cp_reciprocal_tau
            ctx.cp_one_sub_reciprocal_tau = cp_one_sub_reciprocal_tau
            ctx.cp_v_threshold = cp_v_threshold
            ctx.cp_v_reset = cp_v_reset
            ctx.detach_reset = detach_reset
            ctx.sg_cuda_code_fun = sg_cuda_code_fun

        if use_pad:
            return spike_seq[..., :-1], v_v_seq[1:, ..., :-1]
        else:
            return spike_seq, v_v_seq[1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        if ctx.use_pad:
            # grad_spike_seq.shape = [T, N]
            # grad_v_seq.shape = [T, N]
            # h_seq.shape = [T, N + 1]
            # spike_seq.shape = [T, N + 1]
            grad_spike_seq = F.pad(grad_spike_seq, (0, 1))
            grad_v_seq = F.pad(grad_v_seq, (0, 1))

        device = grad_spike_seq.get_device()
        if configure.save_spike_as_bool_in_neuron_kernel:
            spike_seq = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
            h_seq, v_v_seq = ctx.saved_tensors
        else:
            h_seq, spike_seq, v_v_seq = ctx.saved_tensors
        zero_shape = list(grad_spike_seq.shape)
        zero_shape[0] += 1
        zero_data = torch.zeros(zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype)
        grad_x_seq = zero_data[0: -1]
        grad_v_init = zero_data[-1]
        grad_reciprocal_tau = torch.as_tensor(0., device=grad_spike_seq.device, dtype=torch.float32)

        if ctx.cp_v_reset is None:
            hard_reset = False
        else:
            hard_reset = True

        if grad_spike_seq.dtype == torch.float32:
            dtype = 'fp32'
        elif grad_spike_seq.dtype == torch.float16:
            dtype = 'fp16'
        else:
            raise NotImplementedError

        kernel = MultiStepParametricLIFNodePTT.create_bptt_kernel(ctx.sg_cuda_code_fun, ctx.decay_input, hard_reset,
                                                                  ctx.detach_reset, dtype)

        with cuda_utils.DeviceEnvironment(device):

            if hard_reset:
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, grad_reciprocal_tau, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init,
                    grad_reciprocal_tau, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold,
                    ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init,
                               grad_reciprocal_tau, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau,
                               ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel]
            else:
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, grad_reciprocal_tau, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init,
                    grad_reciprocal_tau, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold,
                    ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init,
                               grad_reciprocal_tau, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau,
                               ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel]

            kernel(
                (ctx.blocks,), (ctx.threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )

        if ctx.use_pad:
            return grad_x_seq[..., :-1], grad_v_init[..., :-1], grad_reciprocal_tau, None, None, None, None, None
        else:
            return grad_x_seq, grad_v_init, grad_reciprocal_tau, None, None, None, None, None


def check_multi_step_neuron_output_and_grad(device, multi_step_neuron, shape=[65, 15, 511], *neu_args, **neu_kwargs):
    @torch.no_grad()
    def max_error(x, y):
        return (x - y).abs().max().item()

    def fbptt(m, x: torch.Tensor):
        x = x.detach()
        x.requires_grad_(True)
        spike_seq = m(x)
        (spike_seq * m.v_seq ** 2).sum().backward()
        ret = {
            'spike_seq': spike_seq.detach().clone(),
            'v_seq': m.v_seq.detach().clone(),
            'x.grad': x.grad.clone()
        }
        for i, param in enumerate(m.parameters()):
            ret[f'param_{i}.grad'] = param.grad.detach().clone()
            param.grad.zero_()
        x.grad.zero_()
        m.reset()
        return ret

    for hard_reset in [True, False]:
        for detach_reset in [False, True]:
            for dtype in ['fp32', 'fp16']:
                x = (torch.rand(shape, device=device) - 0.5) * 3.
                if dtype == 'fp16':
                    x = x.half()
                print(f'hard_reset={hard_reset}, detach_reset={detach_reset}, dtype={dtype}')
                model = multi_step_neuron(v_reset=0. if hard_reset else None, detach_reset=detach_reset, *neu_args,
                                          **neu_kwargs)
                # print(model)
                model.to(device)
                if dtype == 'fp16':
                    model = model.half()
                model.backend = 'torch'
                y_torch = fbptt(model, x)

                model.backend = 'cupy'
                y_cupy = fbptt(model, x)

                for key in y_torch.keys():
                    me = max_error(y_torch[key], y_cupy[key])
                    print(key, 'max error', me)
                    if me > 0.5:
                        print(f'y_torch[{key}]={y_torch[key]}, y_cupy[{key}]={y_cupy[key]}')
                print('\n')


def check_single_step_neuron_output_and_grad(device, single_step_neuron, shape=[65, 15, 511], *neu_args, **neu_kwargs):
    @torch.no_grad()
    def max_error(x, y):
        return (x - y).abs().max().item()

    def fbp(m, x: torch.Tensor):
        x = x.detach()
        x.requires_grad_(True)
        T = x.size(0)
        for i in range(T):
            spike = m(x[i])
        (spike * m.v ** 2).sum().backward()
        ret = {
            'spike': spike.detach().clone(),
            'v': m.v.detach().clone(),
            'x.grad': x.grad.clone()
        }
        for i, param in enumerate(m.parameters()):
            ret[f'param_{i}.grad'] = param.grad.detach().clone()
            param.grad.zero_()
        x.grad.zero_()
        m.reset()
        return ret

    for hard_reset in [True, False]:
        for detach_reset in [False, True]:
            for dtype in ['fp32', 'fp16']:
                x = (torch.rand(shape, device=device) - 0.5) * 3.
                if dtype == 'fp16':
                    x = x.half()
                print(f'hard_reset={hard_reset}, detach_reset={detach_reset}, dtype={dtype}')
                model = single_step_neuron(v_reset=0. if hard_reset else None, detach_reset=detach_reset, step_mode='s',
                                           *neu_args,
                                           **neu_kwargs)
                # print(model)
                model.to(device)
                if dtype == 'fp16':
                    model = model.half()
                model.backend = 'torch'
                y_torch = fbp(model, x)

                model.backend = 'cupy'
                y_cupy = fbp(model, x)

                for key in y_torch.keys():
                    me = max_error(y_torch[key], y_cupy[key])
                    print(key, 'max error', me)
                    if me > 0.5:
                        print(f'y_torch[{key}]={y_torch[key]}, y_cupy[{key}]={y_cupy[key]}')
                print('\n')


class MultiStepQIFNodePTT(torch.autograd.Function):
    @staticmethod
    def create_fptt_kernel(hard_reset: bool, dtype: str):
        kernel_name = f'QIFNode_fptt_{"hard" if hard_reset else "soft"}Reset_{dtype}'

        if dtype == 'fp32':
            code = rf'''
            extern "C" __global__
            void {kernel_name}(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
            const float & reciprocal_tau, 
            const float & v_c,
            const float & a0,
            const float & v_threshold,
            const float & v_rest, {'const float & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''
            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < neuron_num)
            {
                const int dt = neuron_num;
                for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                {
                    const int t = index + mem_offset;
                    h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] + a0 * (v_v_seq[t] - v_rest) * (v_v_seq[t] - v_c));
                    if (h_seq[t] >= v_threshold)
                    {
                        spike_seq[t] = 1.0f;
            '''

            if hard_reset:
                code += r'''
                        v_v_seq[t + dt] = v_reset;
                '''
            else:
                code += r'''
                        v_v_seq[t + dt] = h_seq[t] - v_threshold;
                '''

            code += r'''
                    }
                    else
                    {
                        spike_seq[t] = 0.0f;
                        v_v_seq[t + dt] = h_seq[t];
                    }

                }
            }
            }
            '''

        elif dtype == 'fp16':
            code = rf'''
            #include <cuda_fp16.h>
            extern "C" __global__
            void {kernel_name}(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
            const half & reciprocal_tau, 
            const half & v_c,
            const half & a0,
            const half & v_threshold,
            const half & v_rest, {'const half & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel) 
            '''

            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = neuron_num >> 1;
            if (index < stride)
            {
                const int numel_2 = numel >> 1;
                const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                const half2 v_c_half2 = __half2half2(v_c);
                const half2 a0_half2 = __half2half2(a0);
                const half2 v_threshold_half2 = __half2half2(v_threshold);
                const half2 v_rest_half2 = __half2half2(v_rest);
            '''

            if hard_reset:
                code += r'''
                    const half2 v_reset_half2 = __half2half2(v_reset);
                '''

            code += r'''
                for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                {
                    const int t = index + mem_offset;
                    h_seq[t] = __hfma2(__hfma2(__hmul2(__hsub2(v_v_seq[t], v_rest_half2), __hsub2(v_v_seq[t], v_c_half2)), a0_half2, x_seq[t]), reciprocal_tau_half2, v_v_seq[t]);

                    spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
            '''

            if hard_reset:
                code += r'''
                    v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                '''
            else:
                code += r'''
                    v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                '''

            code += r'''
                }
            }
            }
            '''
        else:
            raise TypeError

        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options,
                              backend=configure.cuda_compiler_backend)

    @staticmethod
    def create_bptt_kernel(sg_cuda_code_fun, hard_reset: bool, detach_reset: bool, dtype: str):

        kernel_name = f'QIFNode_bptt_{"hard" if hard_reset else "soft"}Reset_{"detachReset" if detach_reset else ""}_{dtype}'

        code_grad_s_to_h = sg_cuda_code_fun(x='over_th', y='grad_s_to_h', dtype=dtype)

        if dtype == 'fp32':
            code = fr'''
            extern "C" __global__
            void {kernel_name}(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
            float* grad_x_seq, float* grad_v_init,
            const float & a0_over_tau, const float & neg_sum_v_rest_v_c, const float & reciprocal_tau,
            const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''

            code += r'''
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {   
                    float grad_h = 0.0f;  // grad_h will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const float over_th = h_seq[t] - v_threshold;
            '''
            code += code_grad_s_to_h
            if detach_reset:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike_seq[t];
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f;
                    '''
            else:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                    '''

            code += code_grad_v_to_h
            code += r'''
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (1.0f + a0_over_tau * (2.0f * v_v_seq[t + neuron_num] + neg_sum_v_rest_v_c))) * grad_v_to_h;
                grad_x_seq[t] = grad_h * reciprocal_tau;
                }
            grad_v_init[index] = grad_x_seq[index] * (1.0f + a0_over_tau * (2.0f * v_v_seq[index] + neg_sum_v_rest_v_c));
            }
            }
            '''

        elif dtype == 'fp16':
            code = fr'''
            #include <cuda_fp16.h>
            extern "C" __global__
            void {kernel_name}(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
            half2* grad_x_seq, half2* grad_v_init,
            const half & a0_over_tau, const half & neg_sum_v_rest_v_c,
            const half & reciprocal_tau,
            const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''
            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = neuron_num >> 1;
            if (index < stride)
            {   
                const half2 a0_over_tau_half2 = __half2half2(a0_over_tau);
                const half2 neg_sum_v_rest_v_c_half2 = __half2half2(neg_sum_v_rest_v_c);
                const half2 v_threshold_half2 = __half2half2(v_threshold);
                const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
            '''

            if hard_reset:
                code += r'''
                    const half2 v_reset_half2 = __half2half2(v_reset);
                '''

            code += r'''
                half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                {
                    const int t = index + mem_offset;

                    const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
            '''
            code += code_grad_s_to_h

            if detach_reset:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __float2half2_rn(1.0f);
                    '''
            else:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                    '''

            code += code_grad_v_to_h
            code += r'''
                    grad_h = __hfma2(__hfma2(__hfma2(__hfma2(__float2half2_rn(2.0f), v_v_seq[t + stride], neg_sum_v_rest_v_c_half2), a0_over_tau_half2, __float2half2_rn(1.0f)), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));

                    grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                }
            grad_v_init[index] = __hmul2(__hfma2(__hfma2(__float2half2_rn(2.0f), v_v_seq[index], neg_sum_v_rest_v_c_half2), a0_over_tau_half2, __float2half2_rn(1.0f)), grad_x_seq[index]);
            }
            }
            '''
        else:
            raise TypeError
        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options,
                              backend=configure.cuda_compiler_backend)

    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, tau: float, v_threshold: float, v_reset: float,
                v_rest: float, v_c: float, a0: float, detach_reset: bool, sg_cuda_code_fun):
        requires_grad = x_seq.requires_grad or v_init.requires_grad
        device = x_seq.get_device()
        if x_seq.dtype == torch.float32:
            dtype = 'fp32'
            cp_dtype = np.float32
        elif x_seq.dtype == torch.float16:
            dtype = 'fp16'
            cp_dtype = np.half
        else:
            raise NotImplementedError

        use_pad = False
        if dtype == 'fp16' and v_init.numel() % 2 != 0:
            # only fp16 needs even numel because we use half2 to accelerate
            # when numel is odd, we will pad x_seq
            use_pad = True
            x_seq = F.pad(x_seq, (0, 1))  # [T, N] -> [T, N + 1]
            v_init = F.pad(v_init, (0, 1))  # [N] -> [N + 1]

        zero_shape = list(x_seq.shape)
        zero_shape[0] *= 3
        v_seq, h_seq, spike_seq = torch.split(torch.zeros(zero_shape, device=x_seq.device, dtype=x_seq.dtype),
                                              x_seq.shape[0])

        v_v_seq = torch.cat((v_init.unsqueeze(0), v_seq))

        with cuda_utils.DeviceEnvironment(device):
            numel = x_seq.numel()
            neuron_num = numel // x_seq.shape[0]

            threads = configure.cuda_threads
            if dtype == 'fp16':
                assert neuron_num % 2 == 0
                blocks = cuda_utils.cal_blocks(neuron_num >> 1)
                # we will take two neurons to calculate as one neuron in cuda half2
            else:
                blocks = cuda_utils.cal_blocks(neuron_num)

            cp_numel = cupy.asarray(numel)
            cp_neuron_num = cupy.asarray(neuron_num)
            cp_v_threshold = cupy.asarray(v_threshold, dtype=cp_dtype)
            cp_v_rest = cupy.asarray(v_rest, dtype=cp_dtype)
            cp_v_c = cupy.asarray(v_c, dtype=cp_dtype)
            cp_a0 = cupy.asarray(a0, dtype=cp_dtype)
            cp_reciprocal_tau = cupy.asarray(1.0 / tau, dtype=cp_dtype)
            cp_a0_over_tau = cupy.asarray(a0 / tau, dtype=cp_dtype)
            cp_neg_sum_v_rest_v_c = cupy.asarray(-v_rest - v_c, dtype=cp_dtype)

            if v_reset is None:
                cp_v_reset = None
                hard_reset = False
                x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_c, cp_a0, cp_v_threshold, cp_v_rest, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(
                    x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_c, cp_a0, cp_v_threshold, cp_v_rest,
                    cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_c, cp_a0, cp_v_threshold,
                               cp_v_rest, cp_neuron_num, cp_numel]
            else:
                cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                hard_reset = True
                x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_c, cp_a0, cp_v_threshold, cp_v_rest, cp_v_reset, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(
                    x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_c, cp_a0, cp_v_threshold, cp_v_rest,
                    cp_v_reset, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_c, cp_a0, cp_v_threshold,
                               cp_v_rest, cp_v_reset, cp_neuron_num, cp_numel]

            kernel = MultiStepQIFNodePTT.create_fptt_kernel(hard_reset, dtype)

            kernel(
                (blocks,), (threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )

        if requires_grad:
            ctx.use_pad = use_pad
            if configure.save_spike_as_bool_in_neuron_kernel:
                ctx.s_shape = spike_seq.shape
                ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(spike_seq)
                ctx.save_for_backward(h_seq, v_v_seq)
            else:
                ctx.save_for_backward(h_seq, spike_seq, v_v_seq)
            ctx.blocks = blocks
            ctx.threads = threads
            ctx.cp_numel = cp_numel
            ctx.cp_neuron_num = cp_neuron_num
            ctx.cp_a0_over_tau = cp_a0_over_tau
            ctx.cp_neg_sum_v_rest_v_c = cp_neg_sum_v_rest_v_c
            ctx.cp_reciprocal_tau = cp_reciprocal_tau
            ctx.cp_v_threshold = cp_v_threshold
            ctx.cp_v_reset = cp_v_reset
            ctx.detach_reset = detach_reset
            ctx.sg_cuda_code_fun = sg_cuda_code_fun

        if use_pad:
            return spike_seq[..., :-1], v_v_seq[1:, ..., :-1]
        else:
            return spike_seq, v_v_seq[1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        if ctx.use_pad:
            # grad_spike_seq.shape = [T, N]
            # grad_v_seq.shape = [T, N]
            # h_seq.shape = [T, N + 1]
            # spike_seq.shape = [T, N + 1]
            grad_spike_seq = F.pad(grad_spike_seq, (0, 1))
            grad_v_seq = F.pad(grad_v_seq, (0, 1))

        device = grad_spike_seq.get_device()
        if configure.save_spike_as_bool_in_neuron_kernel:
            spike_seq = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
            h_seq, v_v_seq = ctx.saved_tensors
        else:
            h_seq, spike_seq, v_v_seq = ctx.saved_tensors
        zero_shape = list(grad_spike_seq.shape)
        zero_shape[0] += 1
        zero_data = torch.zeros(zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype)
        grad_x_seq = zero_data[0: -1]
        grad_v_init = zero_data[-1]

        if ctx.cp_v_reset is None:
            hard_reset = False
        else:
            hard_reset = True

        if grad_spike_seq.dtype == torch.float32:
            dtype = 'fp32'
        elif grad_spike_seq.dtype == torch.float16:
            dtype = 'fp16'
        else:
            raise NotImplementedError

        kernel = MultiStepQIFNodePTT.create_bptt_kernel(ctx.sg_cuda_code_fun, hard_reset, ctx.detach_reset, dtype)

        with cuda_utils.DeviceEnvironment(device):

            if hard_reset:
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_a0_over_tau, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_a0_over_tau,
                    ctx.cp_neg_sum_v_rest_v_c, ctx.cp_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset,
                    ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init,
                               ctx.cp_a0_over_tau, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_reciprocal_tau, ctx.cp_v_threshold,
                               ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel]
            else:
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_a0_over_tau, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_a0_over_tau,
                    ctx.cp_neg_sum_v_rest_v_c, ctx.cp_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num,
                    ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init,
                               ctx.cp_a0_over_tau, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_reciprocal_tau, ctx.cp_v_threshold,
                               ctx.cp_neuron_num, ctx.cp_numel]

            kernel(
                (ctx.blocks,), (ctx.threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )
        if ctx.use_pad:
            return grad_x_seq[..., :-1], grad_v_init[..., :-1], None, None, None, None, None, None, None, None
        else:
            return grad_x_seq, grad_v_init, None, None, None, None, None, None, None, None


class MultiStepIzhikevichNodePTT(torch.autograd.Function):
    @staticmethod
    def create_fptt_kernel(hard_reset: bool, dtype: str):
        kernel_name = f'IzhikevichNode_fptt_{"hard" if hard_reset else "soft"}Reset_{dtype}'

        if dtype == 'fp32':
            code = rf'''
            extern "C" __global__
            void {kernel_name}(const float* x_seq, float* v_v_seq, float* h_seq, float* w_w_seq, float* spike_seq,
            const float & reciprocal_tau, 
            const float & a0,
            const float & v_c,
            const float & v_threshold,
            const float & v_rest, 
            const float & reciprocal_tau_w,
            const float & a,
            const float & b, {'const float & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''
            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < neuron_num)
            {
                const int dt = neuron_num;
                for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                {
                    const int t = index + mem_offset;
                    h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] + a0 * (v_v_seq[t] - v_rest) * (v_v_seq[t] - v_c) - w_w_seq[t]);
                    const float z = w_w_seq[t] + reciprocal_tau_w * (a * (h_seq[t] - v_rest) - w_w_seq[t]);
                    if (h_seq[t] >= v_threshold)
                    {
                        spike_seq[t] = 1.0f;
            '''

            if hard_reset:
                code += r'''
                        v_v_seq[t + dt] = v_reset;
                '''
            else:
                code += r'''
                        v_v_seq[t + dt] = h_seq[t] - v_threshold;
                '''

            code += r'''
                    }
                    else
                    {
                        spike_seq[t] = 0.0f;
                        v_v_seq[t + dt] = h_seq[t];
                    }
                    w_w_seq[t + dt] = z + b * spike_seq[t];
                }
            }
            }
            '''
        else:
            raise TypeError

        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options,
                              backend=configure.cuda_compiler_backend)

    @staticmethod
    def create_bptt_kernel(sg_cuda_code_fun, hard_reset: bool, detach_reset: bool, dtype: str):

        kernel_name = f'IzhikevichNode_bptt_{"hard" if hard_reset else "soft"}Reset_{"detachReset" if detach_reset else ""}_{dtype}'

        code_grad_s_to_h = sg_cuda_code_fun(x='over_th', y='grad_s_to_h', dtype=dtype)

        if dtype == 'fp32':
            code = fr'''
            extern "C" __global__
            void {kernel_name}(
            const float* grad_spike_seq, const float* grad_v_seq,
            const float* grad_w_seq, const float* h_seq,
            const float* spike_seq, const float* v_v_seq,
            float* grad_x_seq, float* grad_v_init, float* grad_w_init,
            const float & reciprocal_tau, const float & one_sub_reciprocal_tau_w,
            const float & a_over_tau_w, const float & a0_over_tau,
            const float & b, const float & neg_sum_v_rest_v_c,
            const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''

            code += r'''
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {   
                    float grad_h = 0.0f;  // grad_h will be used recursively
                    float grad_w = 0.0f;  // grad_w will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const float over_th = h_seq[t] - v_threshold;
            '''
            code += code_grad_s_to_h
            if detach_reset:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike_seq[t];
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f;
                    '''
            else:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                    '''

            code += code_grad_v_to_h
            code += r'''
                grad_w = -reciprocal_tau * grad_h + one_sub_reciprocal_tau_w * grad_w;
                grad_h = grad_w * (a_over_tau_w + b * grad_s_to_h) + ((1 + a0_over_tau * (2.0f * v_v_seq[t + neuron_num] + neg_sum_v_rest_v_c)) * grad_h + grad_v_seq[t]) * grad_v_to_h + grad_spike_seq[t] * grad_s_to_h;
                grad_x_seq[t] = grad_h * reciprocal_tau;
                }
            grad_v_init[index] = grad_x_seq[index] * (1.0f + a0_over_tau * (2.0f * v_v_seq[index] + neg_sum_v_rest_v_c));
            grad_w_init[index] = -reciprocal_tau * grad_h + one_sub_reciprocal_tau_w * grad_w;

            }
            }
            '''
        else:
            raise TypeError
        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options,
                              backend=configure.cuda_compiler_backend)

    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, w_init: torch.Tensor, tau: float, v_threshold: float,
                v_reset: float, v_rest: float, a: float, b: float, tau_w: float, v_c: float, a0: float,
                detach_reset: bool, sg_cuda_code_fun):
        requires_grad = x_seq.requires_grad or v_init.requires_grad
        device = x_seq.get_device()
        if x_seq.dtype == torch.float32:
            dtype = 'fp32'
            cp_dtype = np.float32
        else:
            raise NotImplementedError

        zero_shape = list(x_seq.shape)
        zero_shape[0] *= 4
        v_seq, h_seq, w_seq, spike_seq = torch.split(torch.zeros(zero_shape, device=x_seq.device, dtype=x_seq.dtype),
                                                     x_seq.shape[0])

        v_v_seq = torch.cat((v_init.unsqueeze(0), v_seq))
        w_w_seq = torch.cat((w_init.unsqueeze(0), w_seq))

        with cuda_utils.DeviceEnvironment(device):
            numel = x_seq.numel()
            neuron_num = numel // x_seq.shape[0]

            threads = configure.cuda_threads

            blocks = cuda_utils.cal_blocks(neuron_num)

            cp_numel = cupy.asarray(numel)
            cp_neuron_num = cupy.asarray(neuron_num)
            cp_v_threshold = cupy.asarray(v_threshold, dtype=cp_dtype)
            cp_v_rest = cupy.asarray(v_rest, dtype=cp_dtype)
            cp_v_c = cupy.asarray(v_c, dtype=cp_dtype)
            cp_a0 = cupy.asarray(a0, dtype=cp_dtype)
            cp_a = cupy.asarray(a, dtype=cp_dtype)
            cp_b = cupy.asarray(b, dtype=cp_dtype)
            cp_reciprocal_tau = cupy.asarray(1. / tau, dtype=cp_dtype)
            cp_reciprocal_tau_w = cupy.asarray(1. / tau_w, dtype=cp_dtype)
            cp_a0_over_tau = cupy.asarray(a0 / tau, dtype=cp_dtype)
            cp_a_over_tau_w = cupy.asarray(a / tau_w, dtype=cp_dtype)
            cp_one_sub_reciprocal_tau_w = cupy.asarray(1. - 1. / tau_w, dtype=cp_dtype)
            cp_neg_sum_v_rest_v_c = cupy.asarray(-v_rest - v_c, dtype=cp_dtype)

            if v_reset is None:
                cp_v_reset = None
                hard_reset = False
                x_seq, v_v_seq, h_seq, w_w_seq, spike_seq, cp_reciprocal_tau, cp_a0, cp_v_c, cp_v_threshold, cp_v_rest, cp_reciprocal_tau_w, cp_a, cp_b, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(
                    x_seq, v_v_seq, h_seq, w_w_seq, spike_seq, cp_reciprocal_tau, cp_a0, cp_v_c, cp_v_threshold,
                    cp_v_rest, cp_reciprocal_tau_w, cp_a, cp_b, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, w_w_seq, spike_seq, cp_reciprocal_tau, cp_a0, cp_v_c,
                               cp_v_threshold, cp_v_rest, cp_reciprocal_tau_w, cp_a, cp_b, cp_neuron_num, cp_numel]
            else:
                cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                hard_reset = True
                x_seq, v_v_seq, h_seq, w_w_seq, spike_seq, cp_reciprocal_tau, cp_a0, cp_v_c, cp_v_threshold, cp_v_rest, cp_reciprocal_tau_w, cp_a, cp_b, cp_v_reset, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(
                    x_seq, v_v_seq, h_seq, w_w_seq, spike_seq, cp_reciprocal_tau, cp_a0, cp_v_c, cp_v_threshold,
                    cp_v_rest, cp_reciprocal_tau_w, cp_a, cp_b, cp_v_reset, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, w_w_seq, spike_seq, cp_reciprocal_tau, cp_a0, cp_v_c,
                               cp_v_threshold, cp_v_rest, cp_reciprocal_tau_w, cp_a, cp_b, cp_v_reset, cp_neuron_num,
                               cp_numel]

            kernel = MultiStepIzhikevichNodePTT.create_fptt_kernel(hard_reset, dtype)

            kernel(
                (blocks,), (threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )

        if requires_grad:
            if configure.save_spike_as_bool_in_neuron_kernel:
                ctx.s_shape = spike_seq.shape
                ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(spike_seq)
                ctx.save_for_backward(h_seq, v_v_seq)
            else:
                ctx.save_for_backward(h_seq, spike_seq, v_v_seq)
            ctx.blocks = blocks
            ctx.threads = threads
            ctx.cp_numel = cp_numel
            ctx.cp_neuron_num = cp_neuron_num
            ctx.cp_reciprocal_tau = cp_reciprocal_tau
            ctx.cp_one_sub_reciprocal_tau_w = cp_one_sub_reciprocal_tau_w
            ctx.cp_a_over_tau_w = cp_a_over_tau_w
            ctx.cp_a0_over_tau = cp_a0_over_tau
            ctx.cp_b = cp_b
            ctx.cp_neg_sum_v_rest_v_c = cp_neg_sum_v_rest_v_c
            ctx.cp_v_threshold = cp_v_threshold
            ctx.cp_v_reset = cp_v_reset
            ctx.detach_reset = detach_reset
            ctx.sg_cuda_code_fun = sg_cuda_code_fun

        return spike_seq, v_v_seq[1:, ], w_w_seq[1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq, grad_w_seq):

        device = grad_spike_seq.get_device()
        if configure.save_spike_as_bool_in_neuron_kernel:
            spike_seq = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
            h_seq, v_v_seq = ctx.saved_tensors
        else:
            h_seq, spike_seq, v_v_seq = ctx.saved_tensors
        zero_shape = list(grad_spike_seq.shape)
        zero_shape[0] += 2
        zero_data = torch.zeros(zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype)
        grad_x_seq = zero_data[0: -2]
        grad_v_init = zero_data[-2]
        grad_w_init = zero_data[-1]

        if ctx.cp_v_reset is None:
            hard_reset = False
        else:
            hard_reset = True

        if grad_spike_seq.dtype == torch.float32:
            dtype = 'fp32'
        else:
            raise NotImplementedError

        kernel = MultiStepIzhikevichNodePTT.create_bptt_kernel(ctx.sg_cuda_code_fun, hard_reset, ctx.detach_reset,
                                                               dtype)

        with cuda_utils.DeviceEnvironment(device):

            if hard_reset:
                grad_spike_seq, grad_v_seq, grad_w_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, grad_w_init, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau_w, ctx.cp_a_over_tau_w, ctx.cp_a0_over_tau, ctx.cp_b, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(
                    grad_spike_seq, grad_v_seq, grad_w_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init,
                    grad_w_init, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau_w, ctx.cp_a_over_tau_w,
                    ctx.cp_a0_over_tau, ctx.cp_b, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_v_threshold, ctx.cp_v_reset,
                    ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, grad_w_seq, h_seq, spike_seq, v_v_seq, grad_x_seq,
                               grad_v_init, grad_w_init, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau_w,
                               ctx.cp_a_over_tau_w, ctx.cp_a0_over_tau, ctx.cp_b, ctx.cp_neg_sum_v_rest_v_c,
                               ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel]
            else:
                grad_spike_seq, grad_v_seq, grad_w_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, grad_w_init, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau_w, ctx.cp_a_over_tau_w, ctx.cp_a0_over_tau, ctx.cp_b, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(
                    grad_spike_seq, grad_v_seq, grad_w_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init,
                    grad_w_init, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau_w, ctx.cp_a_over_tau_w,
                    ctx.cp_a0_over_tau, ctx.cp_b, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_v_threshold, ctx.cp_neuron_num,
                    ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, grad_w_seq, h_seq, spike_seq, v_v_seq, grad_x_seq,
                               grad_v_init, grad_w_init, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau_w,
                               ctx.cp_a_over_tau_w, ctx.cp_a0_over_tau, ctx.cp_b, ctx.cp_neg_sum_v_rest_v_c,
                               ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel]

            kernel(
                (ctx.blocks,), (ctx.threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )
        return grad_x_seq, grad_v_init, grad_w_init, None, None, None, None, None, None, None, None, None, None, None


class MultiStepEIFNodePTT(torch.autograd.Function):
    @staticmethod
    def create_fptt_kernel(hard_reset: bool, dtype: str):
        kernel_name = f'EIFNode_fptt_{"hard" if hard_reset else "soft"}Reset_{dtype}'

        if dtype == 'fp32':
            code = rf'''
            extern "C" __global__
            void {kernel_name}(const float* x_seq, float* v_v_seq, float* h_seq, float* spike_seq,
            const float & reciprocal_tau, 
            const float & delta_T,
            const float & theta_rh,
            const float & v_threshold,
            const float & v_rest, {'const float & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''
            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < neuron_num)
            {
                const int dt = neuron_num;
                for(int mem_offset = 0; mem_offset < numel; mem_offset += neuron_num)
                {
                    const int t = index + mem_offset;
                    h_seq[t] = v_v_seq[t] + reciprocal_tau * (x_seq[t] - v_v_seq[t] + v_rest + delta_T * expf((v_v_seq[t] - theta_rh) / delta_T));
                    if (h_seq[t] >= v_threshold)
                    {
                        spike_seq[t] = 1.0f;
            '''

            if hard_reset:
                code += r'''
                        v_v_seq[t + dt] = v_reset;
                '''
            else:
                code += r'''
                        v_v_seq[t + dt] = h_seq[t] - v_threshold;
                '''

            code += r'''
                    }
                    else
                    {
                        spike_seq[t] = 0.0f;
                        v_v_seq[t + dt] = h_seq[t];
                    }

                }
            }
            }
            '''

        elif dtype == 'fp16':
            code = rf'''
            #include <cuda_fp16.h>
            extern "C" __global__
            void {kernel_name}(const half2* x_seq, half2* v_v_seq, half2* h_seq, half2* spike_seq,
            const half & reciprocal_tau, 
            const half & delta_T,
            const half & theta_rh,
            const half & v_threshold,
            const half & v_rest, {'const half & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel) 
            '''

            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = neuron_num >> 1;
            if (index < stride)
            {
                const int numel_2 = numel >> 1;
                const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                const half2 delta_T_half2 = __half2half2(delta_T);
                const half2 theta_rh_half2 = __half2half2(theta_rh);
                const half2 v_threshold_half2 = __half2half2(v_threshold);
                const half2 v_rest_half2 = __half2half2(v_rest);
            '''

            if hard_reset:
                code += r'''
                    const half2 v_reset_half2 = __half2half2(v_reset);
                '''

            code += r'''
                for(int mem_offset = 0; mem_offset < numel_2; mem_offset += stride)
                {
                    const int t = index + mem_offset;
                    h_seq[t] = __hfma2(__hfma2(h2exp(__h2div(__hsub2(v_v_seq[t], theta_rh_half2), delta_T_half2)), delta_T_half2, __hadd2(__hsub2(x_seq[t], v_v_seq[t]), v_rest_half2)), reciprocal_tau_half2, v_v_seq[t]);

                    spike_seq[t] = __hgeu2(h_seq[t], v_threshold_half2);
            '''

            if hard_reset:
                code += r'''
                    v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], v_reset_half2), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                '''
            else:
                code += r'''
                    v_v_seq[t + stride] = __hadd2(__hmul2(spike_seq[t], __hsub2(h_seq[t], v_threshold_half2)), __hmul2(__hsub2(__float2half2_rn(1.0f), spike_seq[t]), h_seq[t]));
                '''

            code += r'''
                }
            }
            }
            '''
        else:
            raise TypeError

        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options,
                              backend=configure.cuda_compiler_backend)

    @staticmethod
    def create_bptt_kernel(sg_cuda_code_fun, hard_reset: bool, detach_reset: bool, dtype: str):

        kernel_name = f'EIFNode_bptt_{"hard" if hard_reset else "soft"}Reset_{"detachReset" if detach_reset else ""}_{dtype}'

        code_grad_s_to_h = sg_cuda_code_fun(x='over_th', y='grad_s_to_h', dtype=dtype)

        if dtype == 'fp32':
            code = fr'''
            extern "C" __global__
            void {kernel_name}(
            const float* grad_spike_seq, const float* grad_v_seq, const float* h_seq, const float* spike_seq, const float* v_v_seq,
            float* grad_x_seq, float* grad_v_init,
            const float & theta_rh, const float & reciprocal_delta_T,
            const float & reciprocal_tau, const float & one_sub_reciprocal_tau,
            const float & v_threshold, {'const float & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''

            code += r'''
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < neuron_num)
                {   
                    float grad_h = 0.0f;  // grad_h will be used recursively
                    for(int mem_offset = numel - neuron_num; mem_offset >= 0; mem_offset -= neuron_num)
                    {
                        const int t = index + mem_offset;
                        const float over_th = h_seq[t] - v_threshold;
            '''
            code += code_grad_s_to_h
            if detach_reset:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike_seq[t];
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f;
                    '''
            else:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - spike_seq[t] + (v_reset - h_seq[t]) * grad_s_to_h;
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const float grad_v_to_h = 1.0f - v_threshold * grad_s_to_h;
                    '''

            code += code_grad_v_to_h
            code += r'''
                grad_h = grad_spike_seq[t] * grad_s_to_h + (grad_v_seq[t] + grad_h * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[t + neuron_num] - theta_rh) * reciprocal_delta_T))) * grad_v_to_h;
                grad_x_seq[t] = grad_h * reciprocal_tau;
                }
            grad_v_init[index] = grad_x_seq[index] * (one_sub_reciprocal_tau + reciprocal_tau * expf((v_v_seq[index] - theta_rh) * reciprocal_delta_T));
            }
            }
            '''

        elif dtype == 'fp16':
            code = fr'''
            #include <cuda_fp16.h>
            extern "C" __global__
            void {kernel_name}(
            const half2* grad_spike_seq, const half2* grad_v_seq, const half2* h_seq, const half2* spike_seq, const half2* v_v_seq,
            half2* grad_x_seq, half2* grad_v_init,
            const half & theta_rh, const half & reciprocal_delta_T,
            const half & reciprocal_tau, const half & one_sub_reciprocal_tau,
            const half & v_threshold, {'const half & v_reset,' if hard_reset else ''}
            const int & neuron_num, const int & numel)
            '''
            code += r'''
            {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = neuron_num >> 1;
            if (index < stride)
            {   
                const half2 reciprocal_tau_half2 = __half2half2(reciprocal_tau);
                const half2 one_sub_reciprocal_tau_half2 = __half2half2(one_sub_reciprocal_tau);
                const half2 reciprocal_delta_T_half2 = __half2half2(reciprocal_delta_T);
                const half2 theta_rh_half2 = __half2half2(theta_rh);
                const half2 v_threshold_half2 = __half2half2(v_threshold);
            '''

            if hard_reset:
                code += r'''
                    const half2 v_reset_half2 = __half2half2(v_reset);
                '''

            code += r'''
                half2 grad_h = __float2half2_rn(0.0f);  // grad_h will be used recursively
                for(int mem_offset = (numel >> 1) - stride; mem_offset >= 0; mem_offset -= stride)
                {
                    const int t = index + mem_offset;

                    const half2 over_th = __hsub2(h_seq[t], v_threshold_half2);
            '''
            code += code_grad_s_to_h

            if detach_reset:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), spike_seq[t]);
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __float2half2_rn(1.0f);
                    '''
            else:
                if hard_reset:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hfma2(__hsub2(v_reset_half2, h_seq[t]),  grad_s_to_h, __hsub2(__float2half2_rn(1.0f), spike_seq[t]));
                    '''
                else:
                    code_grad_v_to_h = r'''
                    const half2 grad_v_to_h = __hsub2(__float2half2_rn(1.0f), __hmul2(v_threshold_half2, grad_s_to_h));
                    '''

            code += code_grad_v_to_h
            code += r'''
                    grad_h = __hfma2(__hfma2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[t + stride], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_h, grad_v_seq[t]), grad_v_to_h, __hmul2(grad_spike_seq[t], grad_s_to_h));                      
                    grad_x_seq[t] = __hmul2(grad_h, reciprocal_tau_half2);
                }
            grad_v_init[index] = __hmul2(__hfma2(h2exp(__hmul2(__hsub2(v_v_seq[index], theta_rh_half2), reciprocal_delta_T_half2)), reciprocal_tau_half2, one_sub_reciprocal_tau_half2), grad_x_seq[index]);
            }
            }
            '''
        else:
            raise TypeError
        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options,
                              backend=configure.cuda_compiler_backend)

    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, tau: float, v_threshold: float, v_reset: float,
                v_rest: float, theta_rh: float, delta_T: float, detach_reset: bool, sg_cuda_code_fun):
        requires_grad = x_seq.requires_grad or v_init.requires_grad
        device = x_seq.get_device()
        if x_seq.dtype == torch.float32:
            dtype = 'fp32'
            cp_dtype = np.float32
        elif x_seq.dtype == torch.float16:
            dtype = 'fp16'
            cp_dtype = np.half
        else:
            raise NotImplementedError

        use_pad = False
        if dtype == 'fp16' and v_init.numel() % 2 != 0:
            # only fp16 needs even numel because we use half2 to accelerate
            # when numel is odd, we will pad x_seq
            use_pad = True
            x_seq = F.pad(x_seq, (0, 1))  # [T, N] -> [T, N + 1]
            v_init = F.pad(v_init, (0, 1))  # [N] -> [N + 1]

        zero_shape = list(x_seq.shape)
        zero_shape[0] *= 3
        v_seq, h_seq, spike_seq = torch.split(torch.zeros(zero_shape, device=x_seq.device, dtype=x_seq.dtype),
                                              x_seq.shape[0])

        v_v_seq = torch.cat((v_init.unsqueeze(0), v_seq))

        with cuda_utils.DeviceEnvironment(device):
            numel = x_seq.numel()
            neuron_num = numel // x_seq.shape[0]

            threads = configure.cuda_threads
            if dtype == 'fp16':
                assert neuron_num % 2 == 0
                blocks = cuda_utils.cal_blocks(neuron_num >> 1)
                # we will take two neurons to calculate as one neuron in cuda half2
            else:
                blocks = cuda_utils.cal_blocks(neuron_num)

            cp_numel = cupy.asarray(numel)
            cp_neuron_num = cupy.asarray(neuron_num)
            cp_v_threshold = cupy.asarray(v_threshold, dtype=cp_dtype)
            cp_v_rest = cupy.asarray(v_rest, dtype=cp_dtype)
            cp_theta_rh = cupy.asarray(theta_rh, dtype=cp_dtype)
            cp_delta_T = cupy.asarray(delta_T, dtype=cp_dtype)
            cp_reciprocal_delta_T = cupy.asarray(1. / delta_T, dtype=cp_dtype)
            cp_reciprocal_tau = cupy.asarray(1. / tau, dtype=cp_dtype)
            cp_one_sub_reciprocal_tau = cupy.asarray(1. - 1. / tau, dtype=cp_dtype)

            if v_reset is None:
                cp_v_reset = None
                hard_reset = False
                x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_delta_T, cp_theta_rh, cp_v_threshold, cp_v_rest, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(
                    x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_delta_T, cp_theta_rh, cp_v_threshold,
                    cp_v_rest, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_delta_T, cp_theta_rh,
                               cp_v_threshold, cp_v_rest, cp_neuron_num, cp_numel]
            else:
                cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                hard_reset = True
                x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_delta_T, cp_theta_rh, cp_v_threshold, cp_v_rest, cp_v_reset, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(
                    x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_delta_T, cp_theta_rh, cp_v_threshold,
                    cp_v_rest, cp_v_reset, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_delta_T, cp_theta_rh,
                               cp_v_threshold, cp_v_rest, cp_v_reset, cp_neuron_num, cp_numel]

            kernel = MultiStepEIFNodePTT.create_fptt_kernel(hard_reset, dtype)

            kernel(
                (blocks,), (threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )

        if requires_grad:
            ctx.use_pad = use_pad
            if configure.save_spike_as_bool_in_neuron_kernel:
                ctx.s_shape = spike_seq.shape
                ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(spike_seq)
                ctx.save_for_backward(h_seq, v_v_seq)
            else:
                ctx.save_for_backward(h_seq, spike_seq, v_v_seq)
            ctx.blocks = blocks
            ctx.threads = threads
            ctx.cp_numel = cp_numel
            ctx.cp_neuron_num = cp_neuron_num
            ctx.cp_reciprocal_tau = cp_reciprocal_tau
            ctx.cp_one_sub_reciprocal_tau = cp_one_sub_reciprocal_tau
            ctx.cp_theta_rh = cp_theta_rh
            ctx.cp_reciprocal_delta_T = cp_reciprocal_delta_T
            ctx.cp_v_threshold = cp_v_threshold
            ctx.cp_v_reset = cp_v_reset
            ctx.detach_reset = detach_reset
            ctx.sg_cuda_code_fun = sg_cuda_code_fun

        if use_pad:
            return spike_seq[..., :-1], v_v_seq[1:, ..., :-1]
        else:
            return spike_seq, v_v_seq[1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        if ctx.use_pad:
            # grad_spike_seq.shape = [T, N]
            # grad_v_seq.shape = [T, N]
            # h_seq.shape = [T, N + 1]
            # spike_seq.shape = [T, N + 1]
            grad_spike_seq = F.pad(grad_spike_seq, (0, 1))
            grad_v_seq = F.pad(grad_v_seq, (0, 1))

        device = grad_spike_seq.get_device()
        if configure.save_spike_as_bool_in_neuron_kernel:
            spike_seq = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
            h_seq, v_v_seq = ctx.saved_tensors
        else:
            h_seq, spike_seq, v_v_seq = ctx.saved_tensors
        zero_shape = list(grad_spike_seq.shape)
        zero_shape[0] += 1
        zero_data = torch.zeros(zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype)
        grad_x_seq = zero_data[0: -1]
        grad_v_init = zero_data[-1]

        if ctx.cp_v_reset is None:
            hard_reset = False
        else:
            hard_reset = True

        if grad_spike_seq.dtype == torch.float32:
            dtype = 'fp32'
        elif grad_spike_seq.dtype == torch.float16:
            dtype = 'fp16'
        else:
            raise NotImplementedError

        kernel = MultiStepEIFNodePTT.create_bptt_kernel(ctx.sg_cuda_code_fun, hard_reset, ctx.detach_reset, dtype)

        with cuda_utils.DeviceEnvironment(device):

            if hard_reset:
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_theta_rh, ctx.cp_reciprocal_delta_T, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_theta_rh,
                    ctx.cp_reciprocal_delta_T, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold,
                    ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init,
                               ctx.cp_theta_rh, ctx.cp_reciprocal_delta_T, ctx.cp_reciprocal_tau,
                               ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num,
                               ctx.cp_numel]
            else:
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_theta_rh, ctx.cp_reciprocal_delta_T, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(
                    grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_theta_rh,
                    ctx.cp_reciprocal_delta_T, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold,
                    ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init,
                               ctx.cp_theta_rh, ctx.cp_reciprocal_delta_T, ctx.cp_reciprocal_tau,
                               ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel]

            kernel(
                (ctx.blocks,), (ctx.threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )
        if ctx.use_pad:
            return grad_x_seq[..., :-1], grad_v_init[..., :-1], None, None, None, None, None, None, None, None
        else:
            return grad_x_seq, grad_v_init, None, None, None, None, None, None, None, None


def save_cuda_codes(cu_file_path: str = './spikingjelly/activation_based/neuron_kernel_sample.cu'):
    # save all cuda codes to files
    with open(cu_file_path, 'w+') as cu_file:
        cu_file.write('// This file is created by spikingjelly.activation_based.neuron_kernel.save_cuda_codes.\n')
        cu_file.write('// Note that codes in this file will not be executed This file is just created for reading.\n')
        for ms_neu in [MultiStepIFNodePTT]:
            cu_file.write('\n// ' + ms_neu.__name__ + '\n')
            for sg in surrogate._has_cuda_:
                for hard_reset in [True, False]:
                    for dtype in ['fp32', 'fp16']:
                        cu_file.write(
                            f'\n// {ms_neu.__name__} fptt {sg.__name__}, hard_reset={hard_reset}, dtype={dtype}\n')
                        fp_codes = ms_neu.create_fptt_kernel(hard_reset, dtype).code
                        cu_file.write(fp_codes)
                        for detach_reset in [True, False]:
                            cu_file.write(
                                f'\n// {ms_neu.__name__} bptt {sg.__name__}, hard_reset={hard_reset}, dtype={dtype}, detach_reset={detach_reset}\n')
                            bp_codes = ms_neu.create_bptt_kernel(sg().cuda_code, hard_reset, detach_reset,
                                                                 dtype).code
                            cu_file.write(bp_codes)