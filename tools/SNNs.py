import torch
import torch.nn as nn
from .neurons import CUPYPLIFNode
from . import functional, surrogate
import math

class CUPY_SNN_PLIF(nn.Module):
    def __init__(self, in_channels=22, out_num=4, beta=2, w=0.5, surrogate_function=surrogate.Sigmoid(), time_step=128 * 3):
        super(CUPY_SNN_PLIF, self).__init__()
        tau = math.exp(-w) + 1
        channels = int(beta * in_channels)
        kernal = time_step // 32
        self.encode_C = nn.Conv1d(in_channels, channels, kernel_size=(1,), bias=False)
        self.encode_T = nn.Conv1d(channels, channels, kernel_size=(kernal,), padding=(kernal // 2,),
                                  groups=channels, bias=False)
        self.bn_T = nn.BatchNorm1d(channels)
        self.neuron = CUPYPLIFNode(init_tau=tau, surrogate_function=surrogate_function)
        self.Classify = nn.Linear(in_features=channels, out_features=out_num)
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = self.encode_C(x)
        x = self.encode_T(x)
        x = self.bn_T(x).permute(2, 0, 1)
        x = self.neuron(x)
        x = x.mean(0)
        x = self.Classify(x)
        return x