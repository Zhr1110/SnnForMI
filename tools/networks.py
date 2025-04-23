import numpy as np
import torch
import torch.nn as nn
from .neurons import CUPYPLIFNode
from . import layer, functional, surrogate
import torch.nn.functional as F
from typing import Optional
import math

class ShallowConvNet(nn.Module):
    def __init__(self, classes_num, in_channels, time_step, batch_norm=True, batch_norm_alpha=0.1):
        super(ShallowConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.classes_num = classes_num
        n_ch1 = 40
        if self.batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 25), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(in_channels, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5))
        self.layer1.eval()
        out = self.layer1(torch.zeros(1, 1, in_channels, time_step))
        out = torch.nn.functional.avg_pool2d(out, (1, 75), 15)
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time
        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]
        self.clf = nn.Linear(self.n_outputs, self.classes_num)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.layer1(x)
        x = torch.square(x)
        x = torch.nn.functional.avg_pool2d(x, (1, 75), 15)
        x = torch.log(x)
        x = torch.nn.functional.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.clf(x)
        return x

class deepconv(nn.Module):
    def __init__(self, classes_num, in_channels, time_step, batch_norm=True, batch_norm_alpha=0.1, kernal=10):
        super(deepconv, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.classes_num = classes_num
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        self.n_ch4 = 200
        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, kernal), stride=1), # 10 -> 5
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(in_channels, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, kernal), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch2,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, kernal), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch3,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, kernal), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(self.n_ch4,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, kernal), stride=1,bias=False),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(in_channels, 1), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, kernal), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, kernal), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, kernal), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, in_channels, time_step))


        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.n_outputs = out.size()[1]*out.size()[2]*out.size()[3]

        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.classes_num), nn.Dropout(p=0.2))  

    def forward(self, x):
        x = x.unsqueeze(1)  # [N, C, T] -> [N, 1, C, T]
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output=self.clf(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)

class EEGNet(nn.Module):
    def __init__(self,
                 classes_num: int,
                 in_channels: int,
                 time_step: int,
                 kernLenght: int = 64,
                 F1: int = 8,
                 D: int = 2,
                 F2: int = 16,
                 dropout_size: Optional[float] = 0.5,
                ):
        super(EEGNet, self).__init__()
        self.n_classes = classes_num
        self.Chans = in_channels
        self.Samples = time_step
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropout_size
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)), 
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))
        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                      out_features=self.n_classes,
                      bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  
        output = self.block1(x)
        output = self.block2(output)
        output1 = output.reshape(output.size(0), -1)
        output = self.classifier_block(output1)
        return output

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