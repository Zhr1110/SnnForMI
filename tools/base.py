import torch
import torch.nn as nn
import copy
from abc import abstractmethod
from typing import Callable
from tools.surrogate import Sigmoid


try:
    import cupy
except BaseException as e:
    cupy = None

try:
    import lava.lib.dl.slayer as slayer
except BaseException as e:
    slayer = None


def check_backend_library(backend: str):
    """
    * :ref:`API in English <check_backend_library-en>`

    .. _check_backend_library-cn:

    :param backend: ``'torch'``, ``'cupy'`` 或 ``'lava'``
    :type backend: str

    检查某个后端的python库是否已经安装。若未安装则此函数会报错。

    * :ref:`中文 API <check_backend_library-cn>`

    .. _check_backend_library-en:

    :param backend: ``'torch'``, ``'cupy'`` or ``'lava'``
    :type backend: str

    Check whether the python lib for backend is installed. If not, this function will raise an error.
    """
    if backend == 'torch':
        return
    elif backend == 'cupy':
        if cupy is None:
            raise ImportError('CuPy is not installed! You can install it from "https://github.com/cupy/cupy".')
    elif backend == 'lava':
        if slayer is None:
            raise ImportError('Lava-DL is not installed! You can install it from ' \
                              '"https://github.com/lava-nc/lava-dl". ')
    else:
        raise NotImplementedError(backend)


class StepModule:
    def supported_step_mode(self):
        """
        * :ref:`API in English <StepModule.supported_step_mode-en>`

        .. _StepModule.supported_step_mode-cn:

        :return: 包含支持的后端的tuple
        :rtype: tuple[str]

        返回此模块支持的步进模式。

        * :ref:`中文 API <StepModule.supported_step_mode-cn>`

        .. _StepModule.supported_step_mode-en:

        :return: a tuple that contains the supported backends
        :rtype: tuple[str]

        """
        return ('s', 'm')

    @property
    def step_mode(self):
        """
        * :ref:`API in English <StepModule.step_mode-en>`

        .. _StepModule.step_mode-cn:

        :return: 模块当前使用的步进模式
        :rtype: str

        * :ref:`中文 API <StepModule.step_mode-cn>`

        .. _StepModule.step_mode-en:

        :return: the current step mode of this module
        :rtype: str
        """
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        """
        * :ref:`API in English <StepModule.step_mode-setter-en>`

        .. _StepModule.step_mode-setter-cn:

        :param value: 步进模式
        :type value: str

        将本模块的步进模式设置为 ``value``

        * :ref:`中文 API <StepModule.step_mode-setter-cn>`

        .. _StepModule.step_mode-setter-en:

        :param value: the step mode
        :type value: str

        Set the step mode of this module to be ``value``

        """
        if value not in self.supported_step_mode():
            raise ValueError(f'step_mode can only be {self.supported_step_mode()}, but got "{value}"!')
        self._step_mode = value

class SingleModule(StepModule):
    """
    * :ref:`API in English <SingleModule-en>`

    .. _SingleModule-cn:

    只支持单步的模块 (``step_mode == 's'``)。

    * :ref:`中文 API <SingleModule-cn>`

    .. _SingleModule-en:

    The module that only supports for single-step (``step_mode == 's'``)
    """
    def supported_step_mode(self):
        return ('s', )

class MultiStepModule(StepModule):
    """
    * :ref:`API in English <MultiStepModule-en>`

    .. _MultiStepModule-cn:

    只支持多步的模块 (``step_mode == 'm'``)。

    * :ref:`中文 API <MultiStepModule-cn>`

    .. _MultiStepModule-en:

    The module that only supports for multi-step (``step_mode == 'm'``)
    """
    def supported_step_mode(self):
        return ('m', )

class MemoryModule(nn.Module, StepModule):
    def __init__(self):
        """
        * :ref:`API in English <MemoryModule.__init__-en>`

        .. _MemoryModule.__init__-cn:

        ``MemoryModule`` 是SpikingJelly中所有有状态（记忆）模块的基类。

        * :ref:`中文API <MemoryModule.__init__-cn>`

        .. _MemoryModule.__init__-en:

        ``MemoryModule`` is the base class of all stateful modules in SpikingJelly.

        """
        super().__init__()
        self._memories = {}
        self._memories_rv = {}
        self._backend = 'torch'
        self.step_mode = 's'

    @property
    def supported_backends(self):
        """
        * :ref:`API in English <MemoryModule.supported_backends-en>`

        .. _MemoryModule.supported_backends-cn:

        返回支持的后端，默认情况下只有 `('torch', )`

        :return: 支持的后端
        :rtype: tuple[str]

        * :ref:`中文API <MemoryModule.supported_backends-cn>`

        .. _MemoryModule.supported_backends-en:

        Return the supported backends. The default return value is `('torch', )`

        :return: supported backends
        :rtype: tuple[str]

        """
        return ('torch',)

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value: str):
        if value not in self.supported_backends:
            raise NotImplementedError(f'{value} is not a supported backend of {self._get_name()}!')
        check_backend_library(value)
        self._backend = value

    @abstractmethod
    def single_step_forward(self, x: torch.Tensor, *args, **kwargs):
        """
        * :ref:`API in English <MemoryModule.single_step_forward-en>`

        .. _MemoryModule.single_step_forward-cn:

        :param x: input tensor with ``shape = [N, *] ``
        :type x: torch.Tensor

        本模块的单步的前向传播函数


        * :ref:`中文 API <MemoryModule.single_step_forward-cn>`

        .. _MemoryModule.single_step_forward-en:

        :param x: input tensor with ``shape = [N, *] ``
        :type x: torch.Tensor

        The single-step forward function for this module

        """
        pass

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        """
        * :ref:`API in English <MemoryModule.multi_step_forward-en>`

        .. _MemoryModule.multi_step_forward-cn:

        :param x: input tensor with ``shape = [T, N, *] ``
        :type x: torch.Tensor

        本模块的多步的前向传播函数，通过调用 ``T`` 次 ``single_step_forward(x[t], *args, **kwargs)`` 实现


        * :ref:`中文 API <MemoryModule.multi_step_forward-cn>`

        .. _MemoryModule.multi_step_forward-en:

        :param x: input tensor with ``shape = [T, N, *] ``
        :type x: torch.Tensor

        The multi-step forward function for this module, which is implementd by calling ``single_step_forward(x[t], *args, **kwargs)`` over ``T`` times

        """
        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t], *args, **kwargs)
            y_seq.append(y.unsqueeze(0))

        return torch.cat(y_seq, 0)

    def forward(self, *args, **kwargs):
        if self.step_mode == 's':
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return f'step_mode={self.step_mode}, backend={self.backend}'

    def register_memory(self, name: str, value):
        """
        * :ref:`API in English <MemoryModule.register_memory-en>`

        .. _MemoryModule.register_memory-cn:

        :param name: 变量的名字
        :type name: str
        :param value: 变量的值
        :type value: any

        将变量存入用于保存有状态变量（例如脉冲神经元的膜电位）的字典中。这个变量的重置值会被设置为 ``value``。每次调用 ``self.reset()``
        函数后， ``self.name`` 都会被重置为 ``value``。

        * :ref:`中文API <MemoryModule.register_memory-cn>`

        .. _MemoryModule.register_memory-en:

        :param name: variable's name
        :type name: str
        :param value: variable's value
        :type value: any

        Register the variable to memory dict, which saves stateful variables (e.g., the membrane potential of a
        spiking neuron). The reset value of this variable will be ``value``. ``self.name`` will be set to ``value`` after
        each calling of ``self.reset()``.

        """
        assert not hasattr(self, name), f'{name} has been set as a member variable!'
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        """
        * :ref:`API in English <MemoryModule.reset-en>`

        .. _MemoryModule.reset-cn:

        重置所有有状态变量为默认值。

        * :ref:`中文API <MemoryModule.reset-cn>`

        .. _MemoryModule.reset-en:

        Reset all stateful variables to their default values.
        """
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value):
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: str):
        if '_memories' in self.__dict__:
            memories = self.__dict__['_memories']
            if name in memories:
                return memories[name]

        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        _memories = self.__dict__.get('_memories')
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            return super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        memories = list(self._memories.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + memories

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def memories(self):
        """
        * :ref:`API in English <MemoryModule.memories-en>`

        .. _MemoryModule.memories-cn:

        :return: 返回一个所有状态变量的迭代器
        :rtype: Iterator

        * :ref:`中文API <MemoryModule.memories-cn>`

        .. _MemoryModule.memories-en:

        :return: an iterator over all stateful variables
        :rtype: Iterator
        """
        for name, value in self._memories.items():
            yield value

    def named_memories(self):
        """
        * :ref:`API in English <MemoryModule.named_memories-en>`

        .. _MemoryModule.named_memories-cn:

        :return: 返回一个所有状态变量及其名称的迭代器
        :rtype: Iterator

        * :ref:`中文API <MemoryModule.named_memories-cn>`

        .. _MemoryModule.named_memories-en:

        :return: an iterator over all stateful variables and their names
        :rtype: Iterator
        """

        for name, value in self._memories.items():
            yield name, value

    def detach(self):
        """
        * :ref:`API in English <MemoryModule.detach-en>`

        .. _MemoryModule.detach-cn:

        从计算图中分离所有有状态变量。

        .. tip::

            可以使用这个函数实现TBPTT(Truncated Back Propagation Through Time)。


        * :ref:`中文API <MemoryModule.detach-cn>`

        .. _MemoryModule.detach-en:

        Detach all stateful variables.

        .. admonition:: Tip
            :class: tip

            We can use this function to implement TBPTT(Truncated Back Propagation Through Time).

        """

        for key in self._memories.keys():
            if isinstance(self._memories[key], torch.Tensor):
                self._memories[key].detach_()

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)
        # do not apply on default values
        # for key, value in self._memories_rv.items():
        #     if isinstance(value, torch.Tensor):
        #         self._memories_rv[key] = fn(value)
        return super()._apply(fn)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._memories = self._memories.copy()
        return replica


class CodeTyper:
    def __init__(self, indent_num: int):
        """
        :param indent_num: the number of indents
        :type indent_num: int

        A CUDA code formatter with adding indents. The full code can be accessed by ``self.codes``.

        Here is an example:

        .. code-block:: python

            from spikingjelly.activation_based.auto_cuda import base, cfunction

            code0 = cfunction.if_else(z='z', x='x', y='y', mask='mask', dtype='float')
            code1 = cfunction.sigmoid_backward(y='y', x='x', alpha=2., dtype='float')

            codes = ''
            codes += code0
            codes += code1

            print('// Without CodeTyper:')
            print('// ------------------')
            print(codes)
            print('// ------------------')

            ctyper = base.CodeTyper(4)
            ctyper.append(code0)
            ctyper.append(code1)
            print('// With CodeTyper:')
            print('// ------------------')
            print(ctyper.codes)
            print('// ------------------')

        .. code-block:: c++

            // Without CodeTyper:
            // ------------------
            z = x * mask + y * (1.0f - mask);const float sigmoid_backward__sigmoid_ax = 1.0f / (1.0f + expf(- (2.0f) * x));
            y = (1.0f - sigmoid_backward__sigmoid_ax) * sigmoid_backward__sigmoid_ax * (2.0f);
            // ------------------
            // With CodeTyper:
            // ------------------

                z = x * mask + y * (1.0f - mask);
                const float sigmoid_backward__sigmoid_ax = 1.0f / (1.0f + expf(- (2.0f) * x));
                y = (1.0f - sigmoid_backward__sigmoid_ax) * sigmoid_backward__sigmoid_ax * (2.0f);

            // ------------------
        """
        self.indent = ' ' * indent_num
        self.codes = '\n'

    def append(self, codes: str):
        """
        :param codes: cuda codes to be added
        :type codes: str

        Append codes in ``self.codes``.
        """
        codes = codes.replace('\n', '')
        codes = codes.split(';')
        for i in range(codes.__len__()):
            if codes[i].__len__() > 0:
                if codes[i] in ('{', '}'):
                    self.codes += (self.indent + codes[i] + '\n')
                else:
                    self.codes += (self.indent + codes[i] + ';\n')


class CodeBlock:
    def __init__(self, env: CodeTyper):
        """
        :param env: a CodeTyper
        :type env: CodeTyper

        A tool for adding a CUDA code block in ``CodeTyper.code``. It is helpful when we want to calculate by intermediate variables.

        Here is an example:

        .. code-block:: python

            from spikingjelly.activation_based.auto_cuda import base

            ctyper = base.CodeTyper(4)
            with base.CodeBlock(ctyper):
                ctyper.append('// swap x and y')
                ctyper.append('float temp_var = x;')
                ctyper.append('x = y;')
                ctyper.append('y = temp_var;')

            print(ctyper.codes)

        The outputs are:

        .. code-block:: c++

            {
             // swap x and y;
             float temp_var = x;
             x = y;
             y = temp_var;
            }
        """
        self.env = env

    def __enter__(self):
        self.env.append('{')
        self.env.indent += ' '

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.indent = self.env.indent[: -1]
        self.env.append('}')


class BaseNode(MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = Sigmoid(), detach_reset: bool = False,
                 step_mode='s', backend='torch', store_v_seq: bool = False):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()
        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function
        self.step_mode = step_mode
        self.backend = backend
        self.store_v_seq = store_v_seq
        # used in lava_exchange
        self.lava_s_cale = 1 << 6
        # used for cupy backend
        self.forward_kernel = None
        self.backward_kernel = None

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)
        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)