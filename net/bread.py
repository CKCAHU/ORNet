# -*- coding: utf-8 -*-
"""
Hamburger for Pytorch

@author: Gsunshine
"""

from functools import partial
from net import settings
from sync_bn.nn.modules import SynchronizedBatchNorm2d
from torch import nn

# norm_layer = partial(SynchronizedBatchNorm2d, momentum=settings.BN_MOM)
norm_layer = nn.BatchNorm2d


class ConvBNReLU_b(nn.Module):
    @classmethod
    def _same_paddings(cls, kernel_size):
        if kernel_size == 1:
            return 0
        elif kernel_size == 3:
            return 1

    def __init__(self, in_c, out_c,
                 kernel_size=1, stride=1, padding='same',
                 dilation=1, groups=1):
        super().__init__()

        if padding == 'same':
            padding = self._same_paddings(kernel_size)

        self.conv = nn.Conv2d(in_c, out_c,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              groups=groups,
                              bias=False)
        # self.bn = norm_layer(out_c)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.0003, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        return x

