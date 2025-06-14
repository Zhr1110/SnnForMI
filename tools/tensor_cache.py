import torch
import torch.nn.functional as F
import threading
from .auto_cuda import configure, cuda_utils
import logging
try:
    import cupy
except BaseException as e:
    logging.info(f'spikingjelly.activation_based.tensor_cache: {e}')
    cupy = None

class DataTypeConvertCUDACode:
    float2bool = r'''
    extern "C" __global__
            void float2bool(const float* fs, unsigned char* bs, const int &N)
            {
                // assert N == numel / 8 and numel % 8 == 0
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    bs[index] = 0;
                    const int mem_offset = (index << 3);
                    #pragma unroll
                    for(int i = 0; i < 8; i++)
                    {
                        bs[index] += ( ((unsigned char) fs[mem_offset + i]) << i);
                    }
                }
            }
    '''

    half2bool = r'''
    #include <cuda_fp16.h>
    extern "C" __global__
            void half2bool(const half* fs, unsigned char* bs, const int &N)
            {
                // assert N == numel / 8 and numel % 8 == 0
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    bs[index] = 0;
                    const int mem_offset = (index << 3);
                    #pragma unroll
                    for(int i = 0; i < 8; i++)
                    {
                        bs[index] += ( ((unsigned char) __half2float(fs[mem_offset + i])) << i);
                    }
                }
            }
    '''

    bool2float = r'''
    extern "C" __global__
            void bool2float(const unsigned char* bs, float* fs, const int &N)
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    const int mem_offset = (index << 3);
                    unsigned char compressed_v = bs[index];
                    #pragma unroll
                    for(int i = 0; i < 8; i++)
                    {
                        fs[mem_offset + i] = (float) (compressed_v % 2);
                        compressed_v = (compressed_v >> 1);
                    }
                }
            }
    '''

    bool2half = r'''
    #include <cuda_fp16.h>
    extern "C" __global__
            void bool2half(const unsigned char* bs, half* fs, const int &N)
            {
                const int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    const int mem_offset = (index << 3);
                    unsigned char compressed_v = bs[index];
                    #pragma unroll
                    for(int i = 0; i < 8; i++)
                    {
                        fs[mem_offset + i] = __float2half((float) (compressed_v % 2));
                        compressed_v = (compressed_v >> 1);
                    }
                }
            }
    '''
def float_spike_to_bool(spike: torch.Tensor):
    """
    :param spike: a spike tensor whose ``dtype=torch.float`` or ``dtype=torch.half`` and all elements are 0 or 1
    :type spike: torch.Tensor
    :return: (spike_b, s_dtype, s_shape, s_padding)
        spike_b: a compressed spike tensor with ``dtype=torch.uint8`` and each element stores 8 spikes
        s_dtype: the dtype of the original spike
        s_shape: the shape of the original spike
        s_padding: the number of padding elements
    :rtype: tuple

    Compress a float/half spike tensor ``spike`` to an uint8 tensor ``spike_b``. Each element in ``spike_b``
    represents 8 elements of ``spike``.
    """
    s_dtype = spike.dtype
    if s_dtype == torch.float:
        kernel_codes = DataTypeConvertCUDACode.float2bool
        kernel_name = 'float2bool'
    elif s_dtype == torch.half:
        kernel_codes = DataTypeConvertCUDACode.half2bool
        kernel_name = 'half2bool'
    else:
        raise NotImplementedError

    s_shape = spike.shape

    spike = spike.flatten()
    s_padding = 8 - spike.numel() % 8
    if s_padding != 0 and s_padding != 8:
        spike = F.pad(spike, (0, s_padding))
    device_id = spike.get_device()
    spike_b = torch.zeros([spike.numel() // 8], device=spike.device, dtype=torch.uint8)

    if device_id >= 0 and cupy is not None:
        with cuda_utils.DeviceEnvironment(device_id):
            numel = spike_b.numel()
            blocks = cuda_utils.cal_blocks(numel)
            numel = cupy.asarray(numel)
            spike, spike_b, numel = cuda_utils.get_contiguous(spike, spike_b, numel)
            kernel_args = [spike, spike_b, numel]
            kernel = cupy.RawKernel(
                kernel_codes,
                kernel_name,
                options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend
            )
            kernel(
                (blocks,), (configure.cuda_threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device_id,
                    *kernel_args
                )
            )
    else:
        spike = spike.view(-1, 8).to(torch.uint8)
        for i in range(8):
            spike_b += spike[:, i] << i

    return spike_b, s_dtype, s_shape, s_padding

def bool_spike_to_float(spike_b: torch.Tensor, s_dtype: torch.dtype, s_shape: torch.Size, s_padding: int = 0):
    """
    :param spike_b: a compressed spike tensor with ``dtype=torch.uint8`` and each element stores 8 spikes
    :type spike_b: torch.Tensor
    :param s_dtype: the dtype of the original spike
    :type s_dtype: torch.dtype
    :param s_shape: the shape of the original spike
    :type s_shape: torch.Size
    :param s_padding: the number of padding elements
    :type s_padding: int
    :return: the original tensor
    :rtype: torch.Tensor
    """
    device_id = spike_b.get_device()
    spike = torch.zeros(spike_b.numel() * 8, device=spike_b.device, dtype=s_dtype)
    if s_dtype == torch.float:
        kernel_codes = DataTypeConvertCUDACode.bool2float
        kernel_name = 'bool2float'
    elif s_dtype == torch.half:
        kernel_codes = DataTypeConvertCUDACode.bool2half
        kernel_name = 'bool2half'
    else:
        raise NotImplementedError

    if device_id >= 0 and cupy is not None:
        with cuda_utils.DeviceEnvironment(device_id):
            numel = spike_b.numel()
            blocks = cuda_utils.cal_blocks(numel)
            numel = cupy.asarray(numel)
            spike_b, spike, numel = cuda_utils.get_contiguous(spike_b, spike, numel)
            kernel_args = [spike_b, spike, numel]
            kernel = cupy.RawKernel(
                kernel_codes,
                kernel_name,
                options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend
            )
            kernel(
                (blocks,), (configure.cuda_threads,),
                cuda_utils.wrap_args_to_raw_kernel(
                    device_id,
                    *kernel_args
                )
            )
    else:
        spike = spike.view(-1, 8)
        for i in range(8):
            spike[:, i] = spike_b % 2
            spike_b = spike_b >> 1


    if s_padding != 0 and s_padding != 8:
        spike = spike[0: spike.numel() - s_padding]
    return spike.reshape(s_shape)


def tensor_key(x: torch.Tensor):
    x = x.flatten()
    return x.data_ptr(), x[-1].data_ptr(), x.numel()

class BoolTensorCache:
    def __init__(self):
        super().__init__()
        self.cache_dict = {}
        self.cache_refcount_dict = {}
        self.lock = threading.Lock()

    def store_bool(self, spike: torch.FloatTensor or torch.HalfTensor):
        tk = tensor_key(spike)

        self.lock.acquire()
        if tk not in self.cache_dict:
            if configure.save_bool_spike_level == 0:
                self.cache_dict[tk] = (spike.bool(), spike.dtype)
            elif configure.save_bool_spike_level == 1:
                self.cache_dict[tk] = float_spike_to_bool(spike)
            else:
                raise NotImplementedError
            self.cache_refcount_dict[tk] = 1
        else:
            self.cache_refcount_dict[tk] += 1
        self.lock.release()

        return tk

    def get_float(self, tk, spike_shape: torch.Size):
        if configure.save_bool_spike_level == 0:
            spike, s_dtype = self.cache_dict[tk]
            spike = spike.to(s_dtype)
        elif configure.save_bool_spike_level == 1:
            spike = bool_spike_to_float(*self.cache_dict[tk])
        else:
            raise NotImplementedError

        self.lock.acquire()
        self.cache_refcount_dict[tk] -= 1
        if self.cache_refcount_dict[tk] == 0:
            del self.cache_refcount_dict[tk]
            del self.cache_dict[tk]
        self.lock.release()

        return spike.view(spike_shape)


BOOL_TENSOR_CACHE = BoolTensorCache()