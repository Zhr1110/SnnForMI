import logging
from . import base
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable
from torch import Tensor


def reset_net(net: nn.Module):
    """
    * :ref:`API in English <reset_net-en>`

    .. _reset_net-cn:

    :param net: 任何属于 ``nn.Module`` 子类的网络

    :return: None

    将网络的状态重置。做法是遍历网络中的所有 ``Module``，若 ``m `` 为 ``base.MemoryModule`` 函数或者是拥有 ``reset()`` 方法，则调用 ``m.reset()``。

    * :ref:`中文API <reset_net-cn>`

    .. _reset_net-en:

    :param net: Any network inherits from ``nn.Module``

    :return: None

    Reset the whole network.  Walk through every ``Module`` as ``m``, and call ``m.reset()`` if this ``m`` is ``base.MemoryModule`` or ``m`` has ``reset()``.
    """
    for m in net.modules():
        if hasattr(m, 'reset'):
            if not isinstance(m, base.MemoryModule):
                # logging.warning(f'Trying to call `reset()` of {m}, which is not spikingjelly.activation_based.base'
                #                 f'.MemoryModule')
                pass
            m.reset()


def set_step_mode(net: nn.Module, step_mode: str):
    """
    * :ref:`API in English <set_step_mode-en>`

    .. _set_step_mode-cn:

    :param net: 一个神经网络
    :type net: nn.Module
    :param step_mode: 's' (单步模式) 或 'm' (多步模式)
    :type step_mode: str
    :return: None

    将 ``net`` 中所有模块的步进模式设置为 ``step_mode`` 。

    .. note::

        :class:`spikingjelly.activation_based.layer.StepModeContainer`, :class:`spikingjelly.activation_based.layer.ElementWiseRecurrentContainer`,
        :class:`spikingjelly.activation_based.layer.LinearRecurrentContainer` 的子模块（不包含包装器本身）的 ``step_mode`` 不会被改变。


    * :ref:`中文 API <set_step_mode-cn>`

    .. _set_step_mode-en:

    :param net: a network
    :type net: nn.Module
    :param step_mode: 's' (single-step) or 'm' (multi-step)
    :type step_mode: str
    :return: None

    Set ``step_mode`` for all modules in ``net``.

    .. admonition:: Note
        :class: note

        The submodule (not including the container itself) of :class:`spikingjelly.activation_based.layer.StepModeContainer`, :class:`spikingjelly.activation_based.layer.ElementWiseRecurrentContainer`,
        :class:`spikingjelly.activation_based.layer.LinearRecurrentContainer` will not be changed.
    """
    from .layer import StepModeContainer, ElementWiseRecurrentContainer, LinearRecurrentContainer

    keep_step_mode_instance = (
        StepModeContainer, ElementWiseRecurrentContainer, LinearRecurrentContainer
    )
    # step_mode of sub-modules in keep_step_mode_instance will not be changed

    keep_step_mode_containers = []
    for m in net.modules():
        if isinstance(m, keep_step_mode_instance):
            keep_step_mode_containers.append(m)

    for m in net.modules():
        if hasattr(m, 'step_mode'):
            is_contained = False
            for container in keep_step_mode_containers:
                if not isinstance(m, keep_step_mode_instance) and m in container.modules():
                    is_contained = True
                    break
            if is_contained:
                # this function should not change step_mode of submodules in keep_step_mode_containers
                pass
            else:
                # if not isinstance(m, (base.StepModule)):
                #     logging.warning(f'Trying to set the step mode for {m}, which is not spikingjelly.activation_based'
                #                     f'.base.StepModule')
                m.step_mode = step_mode


def multi_step_forward(x_seq: Tensor, single_step_module: nn.Module or list[nn.Module] or tuple[nn.Module] or nn.Sequential or Callable):
    """
    * :ref:`API in English <multi_step_forward-en>`

    .. _multi_step_forward-cn:

    :param x_seq: ``shape=[T, batch_size, ...]`` 的输入tensor
    :type x_seq: Tensor
    :param single_step_module: 一个或多个单步模块
    :type single_step_module: torch.nn.Module or list[nn.Module] or tuple[nn.Module] or torch.nn.Sequential or Callable
    :return: ``shape=[T, batch_size, ...]`` 的输出tensor
    :rtype: torch.Tensor

    在单步模块 ``single_step_module`` 上使用多步前向传播。

    * :ref:`中文 API <multi_step_forward-cn>`

    .. _multi_step_forward-en:

    :param x_seq: the input tensor with ``shape=[T, batch_size, ...]``
    :type x_seq: torch.Tensor
    :param single_step_module: one or many single-step modules
    :type single_step_module: torch.nn.Module or list[nn.Module] or tuple[nn.Module] or torch.nn.Sequential or Callable
    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: torch.torch.Tensor

    Applies multi-step forward on ``single_step_module``.

    """
    y_seq = []
    if isinstance(single_step_module, (list, tuple, nn.Sequential)):
        for t in range(x_seq.shape[0]):
            x_seq_t = x_seq[t]
            for m in single_step_module:
                x_seq_t = m(x_seq_t)
            y_seq.append(x_seq_t)
    else:
        for t in range(x_seq.shape[0]):
            y_seq.append(single_step_module(x_seq[t]))

    return torch.stack(y_seq)


def seq_to_ann_forward(x_seq: Tensor, stateless_module: nn.Module or list or tuple or nn.Sequential or Callable):
    """
    * :ref:`API in English <seq_to_ann_forward-en>`

    .. _seq_to_ann_forward-cn:

    :param x_seq: ``shape=[T, batch_size, ...]`` 的输入tensor
    :type x_seq: Tensor
    :param stateless_module: 单个或多个无状态网络层
    :type stateless_module: torch.nn.Module or list or tuple or torch.nn.Sequential or Callable
    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: Tensor

    * :ref:`中文 API <seq_to_ann_forward-cn>`

    .. _seq_to_ann_forward-en:

    :param x_seq: the input tensor with ``shape=[T, batch_size, ...]``
    :type x_seq: Tensor
    :param stateless_module: one or many stateless modules
    :type stateless_module: torch.nn.Module or list or tuple or torch.nn.Sequential or Callable
    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: Tensor

    Applied forward on stateless modules

    """
    y_shape = [x_seq.shape[0], x_seq.shape[1]]
    y = x_seq.flatten(0, 1)
    if isinstance(stateless_module, (list, tuple, nn.Sequential)):
        for m in stateless_module:
            y = m(y)
    else:
        y = stateless_module(y)
    y_shape.extend(y.shape[1:])
    return y.view(y_shape)