# -*- coding: utf-8 -*-
# @Time : 2023/1/12 17:02 
# @Author : Mingzheng 
# @File : general_net.py
# @desc :



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


class BaseCNN_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dr = nn.Dropout(0.5)

    def forward(self, x):
        # return self.sd(self.bn(F.relu(self.conv(x))))
        return self.dr(self.bn(F.leaky_relu(self.conv(x))))

class GCNN_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dr = nn.Dropout(0.5)

    def forward(self, x):
        # return self.sd(self.bn(F.relu(self.conv(x))))
        return self.dr(self.bn(F.leaky_relu(self.conv(x))))


