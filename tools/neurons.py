import numpy as np
from . import base, surrogate
import torch
from typing import Callable
from .auto_cuda import neuron_kernel, cfunction, cuda_utils, configure
import math
from .auto_cuda.base import CKernel2D
import cupy


class LIFNodeFPTTKernel(neuron_kernel.NeuronFPTTKernel):
    def __init__(self, hard_reset, dtype):
        super().__init__(hard_reset, dtype)
        self.add_param(ctype=f'{dtype} &', cname='w')

    def neuronal_charge(self) -> str:
        # note that v_v_seq[t] is v_seq[t - dt]
        return cfunction.decay_add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', w='w', dtype=self.dtype)


class LIFNodeBPTTKernel(neuron_kernel.NeuronBPTTKernel):
    def __init__(self, surrogate_function, hard_reset, detach_reset, dtype):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        self.add_param(ctype=f'{dtype} &', cname='w')

    def grad_h_next_to_v(self) -> str:
        return cfunction.equal(y=f'const {self.dtype} grad_h_next_to_v', x='w', dtype=self.dtype)

    def grad_h_to_x(self) -> str:
        return cfunction.sub(z=f'const {self.dtype} grad_h_to_x', x='1.0f', y='w', dtype=self.dtype)


class LIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None, w: float,
                forward_kernel: LIFNodeFPTTKernel, backward_kernel: LIFNodeBPTTKernel):
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset,
            'w': w
        }
        requires_grad, blocks, threads, py_dict = neuron_kernel.NeuronATGFBase.pre_forward(py_dict)

        forward_kernel((blocks,), (threads,), py_dict)

        neuron_kernel.NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], blocks=blocks, threads=threads,
                                              numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'],
                                              v_reset=py_dict['v_reset'], w=py_dict['w'], backward_kernel=backward_kernel)
        return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
        backward_kernel, blocks, threads, py_dict = neuron_kernel.NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)
        py_dict.update({'w': ctx.w})
        backward_kernel((blocks,), (threads,), py_dict)
        return py_dict['grad_x_seq'], py_dict['grad_v_init'], None, None, None, None, None


class CUPYLIFNode(base.MemoryModule):
    def __init__(self, surrogate_function: Callable = surrogate.Sigmoid(), v_threshold: float = 1., v_reset: float = 0.,
                 detach_reset: bool = True, backend='torch', w: float = 0.5):
        super().__init__()
        self.register_memory('v', v_reset)
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.step_mode = 'm'
        self.backend = backend
        self.w = w

    def multi_step_forward(self, x_seq: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x_seq[0])
        if x_seq.dtype == torch.float:
            dtype = 'float'
        elif x_seq.dtype == torch.half:
            dtype = 'half2'
        else:
            raise ValueError('x_seq.dtype should be float or half2')
        forward_kernel = LIFNodeFPTTKernel(hard_reset=True, dtype=dtype)
        backward_kernel = LIFNodeBPTTKernel(surrogate_function=self.surrogate_function.cuda_codes, hard_reset=True, detach_reset=self.detach_reset, dtype=dtype)

        # All tensors wil be regard as 2D or 1D. Thus, we use flatten
        spike_seq, v_seq = LIFNodeATGF.apply(x_seq.flatten(1), self.v.flatten(), self.v_threshold, self.v_reset, self.w,
                                             forward_kernel, backward_kernel)
        spike_seq = spike_seq.view(x_seq.shape)
        self.v = v_seq[-1].view(x_seq.shape[1:])
        return spike_seq


### PLIF ###
class ParametricLIFNodeFPTTKernel(neuron_kernel.NeuronFPTTKernel):
    def __init__(self, decay_input: bool, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} *', cname='decay')

    def neuronal_charge(self) -> str:
        if self.hard_reset:
            codes = cfunction.sub(z=f'{self.dtype} LIFNodeFPTTKernel_temp_var', x='v_v_seq[t]', y='v_reset', dtype=self.dtype)
        else:
            codes = f'{self.dtype} LIFNodeFPTTKernel_temp_var = v_v_seq[t];'
        if self.decay_input:
            codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='decay[0]', y='LIFNodeFPTTKernel_temp_var', dtype=self.dtype)
        else:
            codes += cfunction.mul(z='LIFNodeFPTTKernel_temp_var', x='decay[0]', y='LIFNodeFPTTKernel_temp_var',
                                   dtype=self.dtype)
            codes += cfunction.sub(z='LIFNodeFPTTKernel_temp_var', x='x_seq[t]', y='LIFNodeFPTTKernel_temp_var',
                                   dtype=self.dtype)

        codes += cfunction.add(z='h_seq[t]', x='LIFNodeFPTTKernel_temp_var', y='v_v_seq[t]', dtype=self.dtype)

        return codes


class ParametricLIFNodeBPTTKernel(neuron_kernel.NeuronBPTTKernel):
    def __init__(self, decay_input: bool, surrogate_function: Callable, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f'const {dtype} *', cname='decay')
        self.add_param(ctype=f'float *', cname='grad_decay')
        # float to avoid overflow
        self.add_param(ctype=f'const {dtype} *', cname='v_v_seq')

    def grad_h_next_to_v(self) -> str:
        return cfunction.sub(z=f'const {self.dtype} grad_h_next_to_v', x=cfunction.constant(None, x=1., dtype=self.dtype), y='decay[0]', dtype=self.dtype)

    def grad_h_to_x(self) -> str:
        if not self.decay_input:
            return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
        else:
            return f'const {self.dtype} grad_h_to_x = decay[0];'

    @property
    def head(self):
        cuda_threads = 512
        # override
        codes = '''
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
        '''
        codes += fr'''
            
            __shared__ float sdata[{cuda_threads}];
        '''
        # __shared__ float sdata[{configure.cuda_threads}];\
        codes += '''
            if (index < N)
            {
                const int dt = N;
        '''

        codes += self.pre_core

        if self.reverse:
            codes += '''
                for(int t = numel - N + index; t >= 0; t -= dt)
                {
            '''
        else:
            codes += '''
                for(int t = index; t < numel; t += dt)
                {
            '''
        return codes


    @property
    def pre_core(self):
        codes = base.CodeTyper(16)
        # use float to avoid overflow
        codes.append('sdata[threadIdx.x] = 0.0f;')
        return super().pre_core + '\n' + codes.codes

    @property
    def core(self):
        core_codes = base.CodeTyper(18)
        with base.CodeBlock(core_codes):
            if self.decay_input:

                core_codes.append(cfunction.sub(z=f'{self.dtype} temp_var', x='h_seq[t]', y='v_v_seq[t]', dtype=self.dtype))
                core_codes.append(cfunction.mul(z='temp_var', x='temp_var', y='grad_h', dtype=self.dtype))
                core_codes.append(cfunction.div(z='temp_var', x='temp_var', y='decay[0]', dtype=self.dtype))

            else:
                if self.hard_reset:
                    core_codes.append(
                        cfunction.sub(z=f'{self.dtype} temp_var', x='v_reset', y='v_v_seq[t]', dtype=self.dtype))
                    core_codes.append(cfunction.mul(z='temp_var', x='temp_var', y='grad_h', dtype=self.dtype))
                else:
                    core_codes.append(
                        cfunction.mul(z=f'{self.dtype} temp_var', x='grad_h', y='v_v_seq[t]', dtype=self.dtype))
                    core_codes.append(cfunction.neg(y='temp_var', x='temp_var', dtype=self.dtype))


            if self.dtype == 'float':
                core_codes.append('sdata[threadIdx.x] += temp_var;')
            elif self.dtype == 'half2':
                core_codes.append('sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_var), __high2half(temp_var)));')
            else:
                raise NotImplementedError(self.dtype)

        return super().core + '\n' + core_codes.codes

    @property
    def tail(self):
        codes = '''
                }
        '''
        codes += self.post_core
        codes += '''
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
            atomicAdd(grad_decay, sdata[0]);
            }
        }
        '''
        return codes


class ParametricLIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None, decay: torch.Tensor, forward_kernel: ParametricLIFNodeFPTTKernel, backward_kernel: ParametricLIFNodeBPTTKernel):
        if x_seq.dtype == torch.float16 and v_init.numel() % 2 != 0:
            raise ValueError('When using the the PLIF neuron with half2 cupy backend, the numer of neurons should be even to avoid the wrong gradient of tau caused by padding!')
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset,
            'decay': decay,
        }
        requires_grad, blocks, threads, py_dict = neuron_kernel.NeuronATGFBase.pre_forward(py_dict)
        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')
        forward_kernel((blocks,), (threads,), py_dict)
        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None
        neuron_kernel.NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], py_dict['v_v_seq'], blocks=blocks, threads=threads,
                           numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                           backward_kernel=backward_kernel, decay=py_dict['decay'])
        return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
        backward_kernel, blocks, threads, py_dict = neuron_kernel.NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)
        py_dict['decay'] = ctx.decay
        py_dict['grad_decay'] = torch.zeros_like(ctx.decay, dtype=torch.float)
        py_dict['v_v_seq'] = ctx.saved_tensors[1]
        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')
        backward_kernel((blocks,), (threads,), py_dict)
        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None
        return py_dict['grad_x_seq'], py_dict['grad_v_init'], None, None,  py_dict['grad_decay'], None, None


class CUPYPLIFNode(base.MemoryModule):
    def __init__(self, surrogate_function: Callable = surrogate.Sigmoid(), v_threshold: float = 1., v_reset: float = 0.,
                 detach_reset: bool = True, decay_input: bool = True, backend='torch', init_tau: float = 2.0):
        super().__init__()
        self.register_memory('v', v_reset)
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.decay_input = decay_input
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.step_mode = 'm'
        self.backend = backend
        init_w = - math.log(init_tau - 1.)  # tau = e^(-w) + 1
        self.w = torch.as_tensor(init_w).sigmoid().cuda()

    def multi_step_forward(self, x_seq: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x_seq[0])
        if x_seq.dtype == torch.float:
            dtype = 'float'
        elif x_seq.dtype == torch.half:
            dtype = 'half2'
        else:
            raise ValueError('x_seq.dtype should be float or half2')
        forward_kernel = ParametricLIFNodeFPTTKernel(decay_input=self.decay_input, hard_reset=True, dtype=dtype)
        backward_kernel = ParametricLIFNodeBPTTKernel(decay_input=self.decay_input, surrogate_function=self.surrogate_function.cuda_codes, hard_reset=True, detach_reset=self.detach_reset, dtype=dtype)
        # All tensors wil be regard as 2D or 1D. Thus, we use flatten
        spike_seq, v_seq = ParametricLIFNodeATGF.apply(x_seq.flatten(1), self.v.flatten(), self.v_threshold, self.v_reset, self.w, forward_kernel, backward_kernel)
        spike_seq = spike_seq.view(x_seq.shape)
        self.v = v_seq[-1].view(x_seq.shape[1:])
        return spike_seq

class CUPYPLIFNode2(base.MemoryModule):
    def __init__(self, surrogate_function: Callable = surrogate.Sigmoid(), v_threshold: float = 1., v_reset: float = 0.,
                 detach_reset: bool = True, decay_input: bool = True, backend='torch', init_tau: float = 2.0):
        super().__init__()
        self.register_memory('v', v_reset)
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.decay_input = decay_input
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.step_mode = 'm'
        self.backend = backend
        init_w = - math.log(init_tau - 1.)  # tau = e^(-w) + 1
        self.w = torch.as_tensor(init_w).sigmoid().cuda()

    def multi_step_forward(self, x_seq: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x_seq[0])
        if x_seq.dtype == torch.float:
            dtype = 'float'
        elif x_seq.dtype == torch.half:
            dtype = 'half2'
        else:
            raise ValueError('x_seq.dtype should be float or half2')
        forward_kernel = ParametricLIFNodeFPTTKernel(decay_input=self.decay_input, hard_reset=True, dtype=dtype)
        backward_kernel = ParametricLIFNodeBPTTKernel(decay_input=self.decay_input, surrogate_function=self.surrogate_function.cuda_codes, hard_reset=True, detach_reset=self.detach_reset, dtype=dtype)
        # All tensors wil be regard as 2D or 1D. Thus, we use flatten
        spike_seq, v_seq = ParametricLIFNodeATGF.apply(x_seq.flatten(1), self.v.flatten(), self.v_threshold, self.v_reset, self.w, forward_kernel, backward_kernel)
        spike_seq = spike_seq.view(x_seq.shape)
        self.v = v_seq[-1].view(x_seq.shape[1:])
        return spike_seq

def neuronal_hard_reset(v_next: str, h: str, spike: str, v_reset: str, dtype: str = 'float'):
    if dtype == 'float':
        return f'{v_next} = {h} * (1.0f - {spike}) + {v_reset} * {spike};'
    elif dtype == 'half2':
        return f'{v_next} = __hfma2({h}, __hsub2(__float2half2_rn(1.0f), {spike}), __hmul2(v_reset, {spike}));'
    else:
        raise NotImplementedError(dtype)

def neuronal_soft_reset(v_next: str, h: str, spike: str, v_th: str, dtype: str = 'float'):
    if dtype == 'float':
        return f'{v_next} = {h} - {v_th} * {spike};'
    elif dtype == 'half2':
        return f'{v_next} = __hsub2({h}, __hmul2({v_th}, {spike}));'
    else:
        raise NotImplementedError(dtype)


def neuronal_fire(spike: str, v: str, v_th: str, dtype: str = 'float'):
    if dtype == 'float':
        return cfunction.heaviside(y=spike, x=f'({v} - {v_th})', dtype=dtype)
    elif dtype == 'half2':
        return cfunction.heaviside(y=spike, x=f'__hsub2({v}, {v_th})', dtype=dtype)
    else:
        raise NotImplementedError(dtype)

def neuronal_i_reset(i_next: str, i: str, spike: str, dtype: str = 'float'):
    if dtype == 'float':
        return f'{i_next} = {i} * (1.0f - {spike});'
    elif dtype == 'half2':
        return f'{i_next} = __hmul2({i}, __hsub2(__float2half2_rn(1.0f), {spike}));'
    else:
        raise NotImplementedError(dtype)


class IFNodeFPTTKernel(neuron_kernel.NeuronFPTTKernel):
    def neuronal_charge(self) -> str:
        return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)


class IFNodeBPTTKernel(neuron_kernel.NeuronBPTTKernel):
    def grad_h_next_to_v(self) -> str:
        return cfunction.constant(y=f'const {self.dtype} grad_h_next_to_v', x=1., dtype=self.dtype)

    def grad_h_to_x(self) -> str:
        return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)


class IFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None,
                forward_kernel: IFNodeFPTTKernel, backward_kernel: IFNodeBPTTKernel):
        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset
        }
        requires_grad, blocks, threads, py_dict = neuron_kernel.NeuronATGFBase.pre_forward(py_dict)

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        forward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        neuron_kernel.NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], blocks=blocks, threads=threads,
                                numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'],
                                v_reset=py_dict['v_reset'],
                                backward_kernel=backward_kernel)

        return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):

        backward_kernel, blocks, threads, py_dict = neuron_kernel.NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)

        if py_dict['v_reset'] is None:
            py_dict.pop('v_reset')

        backward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict:
            py_dict['v_reset'] = None

        return py_dict['grad_x_seq'], py_dict['grad_v_init'], None, None, None, None


class CUPYIFNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float or None = 0.,
                surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.detach_reset = detach_reset
        self.step_mode = 'm'
        if v_reset is not None:
            self.register_memory('v', v_reset)
        else:
            self.register_memory('v', 0.)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x_seq[0])
        hard_reset = self.v_reset is not None
        if x_seq.dtype == torch.float:
            dtype = 'float'
        elif x_seq.dtype == torch.half:
            dtype = 'half2'
        forward_kernel = IFNodeFPTTKernel(hard_reset=hard_reset, dtype=dtype)
        backward_kernel = IFNodeBPTTKernel(surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset, detach_reset=self.detach_reset, dtype=dtype)
        # All tensors wil be regard as 2D or 1D. Thus, we use flatten
        spike_seq, v_seq = IFNodeATGF.apply(x_seq.flatten(1), self.v.flatten(), self.v_threshold, self.v_reset, forward_kernel, backward_kernel)

        spike_seq = spike_seq.view(x_seq.shape)
        self.v = v_seq[-1].view(x_seq.shape[1:])

        return spike_seq


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

        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)


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
        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)


    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, tau: float, v_threshold: float, v_reset: float, v_rest: float, v_c: float, a0: float, detach_reset: bool, sg_cuda_code_fun):
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
        v_seq, h_seq, spike_seq = torch.split(torch.zeros(zero_shape, device=x_seq.device, dtype=x_seq.dtype), x_seq.shape[0])

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
                x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_c, cp_a0, cp_v_threshold, cp_v_rest, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_c, cp_a0, cp_v_threshold, cp_v_rest, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_c, cp_a0, cp_v_threshold, cp_v_rest, cp_neuron_num, cp_numel]
            else:
                cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                hard_reset = True
                x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_c, cp_a0, cp_v_threshold, cp_v_rest, cp_v_reset, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_c, cp_a0, cp_v_threshold, cp_v_rest, cp_v_reset, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_v_c, cp_a0, cp_v_threshold, cp_v_rest, cp_v_reset, cp_neuron_num, cp_numel]

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
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_a0_over_tau, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_a0_over_tau, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_a0_over_tau, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel]
            else:
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_a0_over_tau, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_a0_over_tau, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_a0_over_tau, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel]

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

class CUPYQIFNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = -0.1,
                 detach_reset: bool = True, tau: float = 2.0, v_rest: float = 0.,
                 v_c: float=0.8, a0: float=1.0):
        super().__init__()
        self.register_memory('v', v_reset)
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.v_reset = v_reset
        self.tau = tau
        self.v_c = v_c
        self.v_rest = v_rest
        self.a0 = a0
        self.step_mode = 'm'

    def multi_step_forward(self, x_seq: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x_seq[0])
        if x_seq.dtype != torch.float and x_seq.dtype != torch.half:
            raise ValueError('x_seq.dtype should be float or half2')
        # All tensors wil be regard as 2D or 1D. Thus, we use flatten
        spike_seq, v_seq = MultiStepQIFNodePTT.apply(x_seq.flatten(1), self.v.flatten(), self.tau, self.v_threshold, self.v_reset, self.v_rest, self.v_c, self.a0, self.detach_reset, cfunction.sigmoid2)
        spike_seq = spike_seq.view(x_seq.shape)
        self.v = v_seq[-1].view(x_seq.shape[1:])
        return spike_seq


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

        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)


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
        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)


    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, tau: float, v_threshold: float, v_reset: float, v_rest: float, theta_rh: float, delta_T: float, detach_reset: bool, sg_cuda_code_fun):
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
        v_seq, h_seq, spike_seq = torch.split(torch.zeros(zero_shape, device=x_seq.device, dtype=x_seq.dtype), x_seq.shape[0])

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
            cp_reciprocal_tau = cupy.asarray(1./tau, dtype=cp_dtype)
            cp_one_sub_reciprocal_tau = cupy.asarray(1. - 1./tau, dtype=cp_dtype)

            if v_reset is None:
                cp_v_reset = None
                hard_reset = False
                x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_delta_T, cp_theta_rh, cp_v_threshold, cp_v_rest, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_delta_T, cp_theta_rh, cp_v_threshold, cp_v_rest, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_delta_T, cp_theta_rh, cp_v_threshold, cp_v_rest, cp_neuron_num, cp_numel]
            else:
                cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                hard_reset = True
                x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_delta_T, cp_theta_rh, cp_v_threshold, cp_v_rest, cp_v_reset, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_delta_T, cp_theta_rh, cp_v_threshold, cp_v_rest, cp_v_reset, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, spike_seq, cp_reciprocal_tau, cp_delta_T, cp_theta_rh, cp_v_threshold, cp_v_rest, cp_v_reset, cp_neuron_num, cp_numel]

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
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_theta_rh, ctx.cp_reciprocal_delta_T, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_theta_rh, ctx.cp_reciprocal_delta_T, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_theta_rh, ctx.cp_reciprocal_delta_T, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel]
            else:
                grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_theta_rh, ctx.cp_reciprocal_delta_T, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_theta_rh, ctx.cp_reciprocal_delta_T, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, ctx.cp_theta_rh, ctx.cp_reciprocal_delta_T, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel]

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
    

class CUPYEIFNode(base.MemoryModule):
    def __init__(self, tau: float = 2., delta_T: float = 1., theta_rh: float = 0.8, v_threshold: float = 1.,
                 v_rest: float = 0., v_reset: float = -0.1, detach_reset: bool = False):
        super().__init__()
        self.register_memory('v', v_reset)
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.v_reset = v_reset
        self.tau = tau
        self.delta_T = delta_T
        self.v_rest = v_rest
        self.theta_rh = theta_rh
        self.step_mode = 'm'

    def multi_step_forward(self, x_seq: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x_seq[0])
        if x_seq.dtype != torch.float and x_seq.dtype != torch.half:
            raise ValueError('x_seq.dtype should be float or half2')
        # All tensors wil be regard as 2D or 1D. Thus, we use flatten
        spike_seq, v_seq = MultiStepEIFNodePTT.apply(x_seq.flatten(1), self.v.flatten(), self.tau, self.v_threshold, self.v_reset, self.v_rest, self.theta_rh, self.delta_T, self.detach_reset, cfunction.sigmoid2)
        spike_seq = spike_seq.view(x_seq.shape)
        self.v = v_seq[-1].view(x_seq.shape[1:])
        return spike_seq


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

        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)


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
        return cupy.RawKernel(code, kernel_name, options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)


    @staticmethod
    def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, w_init: torch.Tensor, tau: float, v_threshold: float, v_reset: float, v_rest: float, a: float, b: float, tau_w: float, v_c: float, a0: float, detach_reset: bool, sg_cuda_code_fun):
        requires_grad = x_seq.requires_grad or v_init.requires_grad
        device = x_seq.get_device()
        if x_seq.dtype == torch.float32:
            dtype = 'fp32'
            cp_dtype = np.float32
        else:
            raise NotImplementedError

        zero_shape = list(x_seq.shape)
        zero_shape[0] *= 4
        v_seq, h_seq, w_seq, spike_seq = torch.split(torch.zeros(zero_shape, device=x_seq.device, dtype=x_seq.dtype), x_seq.shape[0])

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
            cp_reciprocal_tau_w = cupy.asarray(1./ tau_w, dtype=cp_dtype)
            cp_a0_over_tau = cupy.asarray(a0 / tau, dtype=cp_dtype)
            cp_a_over_tau_w = cupy.asarray(a / tau_w, dtype=cp_dtype)
            cp_one_sub_reciprocal_tau_w = cupy.asarray(1. - 1./tau_w, dtype=cp_dtype)
            cp_neg_sum_v_rest_v_c = cupy.asarray(-v_rest - v_c, dtype=cp_dtype)

            if v_reset is None:
                cp_v_reset = None
                hard_reset = False
                x_seq, v_v_seq, h_seq, w_w_seq, spike_seq, cp_reciprocal_tau, cp_a0, cp_v_c, cp_v_threshold, cp_v_rest, cp_reciprocal_tau_w, cp_a, cp_b, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(x_seq, v_v_seq, h_seq, w_w_seq, spike_seq, cp_reciprocal_tau, cp_a0, cp_v_c, cp_v_threshold, cp_v_rest, cp_reciprocal_tau_w, cp_a, cp_b, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, w_w_seq, spike_seq, cp_reciprocal_tau, cp_a0, cp_v_c, cp_v_threshold, cp_v_rest, cp_reciprocal_tau_w, cp_a, cp_b, cp_neuron_num, cp_numel]
            else:
                cp_v_reset = cupy.asarray(v_reset, dtype=cp_dtype)
                hard_reset = True
                x_seq, v_v_seq, h_seq, w_w_seq, spike_seq, cp_reciprocal_tau, cp_a0, cp_v_c, cp_v_threshold, cp_v_rest, cp_reciprocal_tau_w, cp_a, cp_b, cp_v_reset, cp_neuron_num, cp_numel = cuda_utils.get_contiguous(x_seq, v_v_seq, h_seq, w_w_seq, spike_seq, cp_reciprocal_tau, cp_a0, cp_v_c, cp_v_threshold, cp_v_rest, cp_reciprocal_tau_w, cp_a, cp_b, cp_v_reset, cp_neuron_num, cp_numel)
                kernel_args = [x_seq, v_v_seq, h_seq, w_w_seq, spike_seq, cp_reciprocal_tau, cp_a0, cp_v_c, cp_v_threshold, cp_v_rest, cp_reciprocal_tau_w, cp_a, cp_b, cp_v_reset, cp_neuron_num, cp_numel]

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

        kernel = MultiStepIzhikevichNodePTT.create_bptt_kernel(ctx.sg_cuda_code_fun, hard_reset, ctx.detach_reset, dtype)

        with cuda_utils.DeviceEnvironment(device):

            if hard_reset:
                grad_spike_seq, grad_v_seq, grad_w_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, grad_w_init, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau_w, ctx.cp_a_over_tau_w, ctx.cp_a0_over_tau, ctx.cp_b, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(grad_spike_seq, grad_v_seq, grad_w_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, grad_w_init, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau_w, ctx.cp_a_over_tau_w, ctx.cp_a0_over_tau, ctx.cp_b, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, grad_w_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, grad_w_init, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau_w, ctx.cp_a_over_tau_w, ctx.cp_a0_over_tau, ctx.cp_b, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_v_threshold, ctx.cp_v_reset, ctx.cp_neuron_num, ctx.cp_numel]
            else:
                grad_spike_seq, grad_v_seq, grad_w_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, grad_w_init, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau_w, ctx.cp_a_over_tau_w, ctx.cp_a0_over_tau, ctx.cp_b, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel = cuda_utils.get_contiguous(grad_spike_seq, grad_v_seq, grad_w_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, grad_w_init, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau_w, ctx.cp_a_over_tau_w, ctx.cp_a0_over_tau, ctx.cp_b, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel)
                kernel_args = [grad_spike_seq, grad_v_seq, grad_w_seq, h_seq, spike_seq, v_v_seq, grad_x_seq, grad_v_init, grad_w_init, ctx.cp_reciprocal_tau, ctx.cp_one_sub_reciprocal_tau_w, ctx.cp_a_over_tau_w, ctx.cp_a0_over_tau, ctx.cp_b, ctx.cp_neg_sum_v_rest_v_c, ctx.cp_v_threshold, ctx.cp_neuron_num, ctx.cp_numel]

            kernel(
                (ctx.blocks,), (ctx.threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device,
                    *kernel_args
                )
            )
        return grad_x_seq, grad_v_init, grad_w_init, None, None, None, None, None, None, None, None, None, None, None



class CUPYIzhikevichNode(base.MemoryModule):
    def __init__(self, u_threshold: float = 1., tau: float = 2., detach_reset: bool = True, tau_w: float = 2., v_c: float = 0.8,
                 k: float = 1, a: float = 1., b: float = 0.02, c: float = 0., d: float = 0.2, ur: float = -0.05, v_threshold: float = 1.0):
        super().__init__()
        self.register_memory('v', c)
        self.register_memory('u', c)
        self.v_threshold = v_threshold
        self.v_c = v_c
        self.detach_reset = detach_reset
        self.v_reset = c
        self.a0 = k
        self.a = a * b
        self.b = d
        self.v_rest = ur
        self.tau = tau
        self.tau_w = tau_w
        self.step_mode = 'm'

    def multi_step_forward(self, x_seq: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.zeros_like(x_seq[0])
        if isinstance(self.u, float):
            self.u = torch.zeros_like(x_seq[0])
        if x_seq.dtype != torch.float and x_seq.dtype != torch.half:
            raise ValueError('x_seq.dtype should be float or half2')
        # All tensors wil be regard as 2D or 1D. Thus, we use flatten
        spike_seq, v_seq, u_seq = MultiStepIzhikevichNodePTT.apply(x_seq.flatten(1), self.v.flatten(), self.u.flatten(), self.tau, self.v_threshold, self.v_reset, self.v_rest, self.a, self.b, self.tau_w, self.v_c, self.a0, self.detach_reset, cfunction.sigmoid2)
        spike_seq = spike_seq.view(x_seq.shape)
        self.v = v_seq[-1].view(x_seq.shape[1:])
        self.u = u_seq[-1].view(x_seq.shape[1:])
        return spike_seq
    
