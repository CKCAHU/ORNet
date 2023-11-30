import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, fftshift, ifft2, ifftshift
from config import getConfig
import numpy as np
from net.conv_modules import BasicConv2d, DWConv, DWSConv

cfg = getConfig()


class Fourier_Edge_extractor(nn.Module):
    def __init__(self, radius, channel):
        super(Fourier_Edge_extractor, self).__init__()
        self.radius = radius
        # self.UAM = UnionAttentionModule(channel, only_channel_tracing=True)

        # DWS + DWConv
        self.DWSConv = DWSConv(channel, channel, kernel=3, padding=1, kernels_per_layer=1)
        self.DWConv1 = nn.Sequential(
            DWConv(channel, channel, kernel=1, padding=0, dilation=1),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv2 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=1, dilation=1),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv3 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=3, dilation=3),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.DWConv4 = nn.Sequential(
            DWConv(channel, channel, kernel=3, padding=5, dilation=5),
            BasicConv2d(channel, channel // 4, 1),
        )
        self.conv = BasicConv2d(channel, 1, 1)

    def distance(self, i, j, imageSize, r):
        dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
        if dis < r:
            return 1.0
        else:
            return 0

    def mask_radial(self, img, r):
        batch, channels, rows, cols = img.shape
        mask = torch.zeros((rows, cols), dtype=torch.float32)
        for i in range(rows):
            for j in range(cols):
                mask[i, j] = self.distance(i, j, imageSize=rows, r=r)
        return mask

    def forward(self, x):
        """
        Input:
            The first encoder block representation: (B, C, H, W)
        Returns:
            Edge refined representation: X + edge (B, C, H, W)
        """
        x_fft = fft2(x, dim=(-2, -1))
        x_fft = fftshift(x_fft)

        # Mask -> low, high separate
        mask = self.mask_radial(img=x, r=self.radius).cuda()
        high_frequency = x_fft * (1 - mask)
        x_fft = ifftshift(high_frequency)
        x_fft = ifft2(x_fft, dim=(-2, -1))
        x_H = torch.abs(x_fft)

        # x_H, _ = self.UAM.Channel_Tracer(x_H)
        edge_maks = self.DWSConv(x_H)
        skip = edge_maks.clone()

        edge_maks = torch.cat([self.DWConv1(edge_maks), self.DWConv2(edge_maks),
                               self.DWConv3(edge_maks), self.DWConv4(edge_maks)], dim=1) + skip
        edge = torch.relu(self.conv(edge_maks))

        # x = x + edge  # Feature + Masked Edge information

        return self.conv(x + edge)


class GlobalAvgPool(nn.Module):
    def __init__(self, flatten=False):
        super(GlobalAvgPool, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class Channel_Attention(nn.Module):
    def __init__(self, n_channels, only_channel_tracing=False):
        super(Channel_Attention, self).__init__()
        self.GAP = GlobalAvgPool()
        self.confidence_ratio = cfg.gamma
        self.bn = nn.BatchNorm2d(n_channels)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.Dropout3d(self.confidence_ratio)
        )
        self.channel_q = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.channel_k = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.channel_v = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)

        self.fc = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, stride=1,
                            padding=0, bias=False)

        if not only_channel_tracing:
            self.spatial_q = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
            self.spatial_k = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
            self.spatial_v = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1,
                                       padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def masking(self, x, mask):
        mask = mask.squeeze(3).squeeze(2)
        threshold = torch.quantile(mask, self.confidence_ratio, dim=-1, keepdim=True)
        mask[mask <= threshold] = 0.0
        mask = mask.unsqueeze(2).unsqueeze(3)
        mask = mask.expand(-1, x.shape[1], x.shape[2], x.shape[3]).contiguous()
        masked_x = x * mask

        return masked_x

    def Channel_Tracer(self, x):
        avg_pool = self.GAP(x)
        x_norm = self.norm(avg_pool)

        q = self.channel_q(x_norm).squeeze(-1)
        k = self.channel_k(x_norm).squeeze(-1)
        v = self.channel_v(x_norm).squeeze(-1)

        # softmax(Q*K^T)
        QK_T = torch.matmul(q, k.transpose(1, 2))
        alpha = F.softmax(QK_T, dim=-1)

        # a*v
        att = torch.matmul(alpha, v).unsqueeze(-1)
        att = self.fc(att)
        att = self.sigmoid(att)

        output = (x * att) + x
        alpha_mask = att.clone()

        return output, alpha_mask

    def forward(self, x):
        X_c, alpha_mask = self.Channel_Tracer(x)
        X_c = self.bn(X_c)
        x_drop = self.masking(X_c, alpha_mask)

        q = self.spatial_q(x_drop).squeeze(1)
        k = self.spatial_k(x_drop).squeeze(1)
        v = self.spatial_v(x_drop).squeeze(1)

        # softmax(Q*K^T)
        QK_T = torch.matmul(q, k.transpose(1, 2))
        alpha = F.softmax(QK_T, dim=-1)

        output = torch.matmul(alpha, v).unsqueeze(1) + v.unsqueeze(1)

        return output


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s


class CBR(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
                 bias=True, use_relu=True, use_bn=True, frozen=False, residual=False):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=bias)
        self.residual = residual
        if use_bn:
            self.bn = nn.BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x1 = self.conv(x)
        if self.bn is not None:
            x1 = self.bn(x1)
        if self.residual and x1.shape[1] == x.shape[1]:
            x1 = x + x1
        if self.act is not None:
            x1 = self.act(x1)

        return x1


class ReceptiveConv(nn.Module):
    def __init__(self, in_channel, out_channel, baseWidth=24, scale=4, dilation=None, aggregation=True,
                 use_dwconv=False):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(ReceptiveConv, self).__init__()
        assert scale >= 1, 'The input scale must be a positive value'

        self.width = int(math.floor(out_channel * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(in_channel, self.width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width * scale)
        # self.nums = 1 if scale == 1 else scale - 1
        self.nums = scale

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dilation = [1] * self.nums if dilation is None else dilation
        for i in range(self.nums):
            self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3,
                                        padding=dilation[i], dilation=dilation[i],bias=False))
            self.bns.append(nn.BatchNorm2d(self.width))

        self.conv3 = nn.Conv2d(self.width * scale, out_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        self.aggregation = aggregation

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if self.aggregation:
                sp = spx[i] if i == 0 else sp + spx[i]
            else:
                sp = spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = sp if i == 0 else torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x
        out = self.relu(out)

        return out


class Object_Refinement_module(nn.Module):
    def __init__(self, dim, expand=2, scale=4):
        super(Object_Refinement_module, self).__init__()
        self.width = int(dim * expand / scale)
        self.act = nn.ReLU()
        self.channel_expand = nn.Sequential(
            nn.Conv2d(dim, dim * expand, kernel_size=(1, 1), groups=dim, bias=False),
            nn.BatchNorm2d(dim * expand))

        self.conv0_0 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(3, 3), padding=(1, 1), dilation=(1, 1), bias=False),
            nn.BatchNorm2d(self.width))
        self.conv0_1 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(1, 5), padding=(0, 2), groups=self.width, bias=False),
            nn.Conv2d(self.width, self.width, kernel_size=(5, 1), padding=(2, 0), groups=self.width, bias=False),
            nn.BatchNorm2d(self.width))
        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * self.width, self.width, kernel_size=(1, 1), groups=self.width, bias=False),
            nn.BatchNorm2d(self.width))

        self.conv1_0 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(3, 3), padding=(2, 2), dilation=(2, 2), bias=False),
            nn.BatchNorm2d(self.width))
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(1, 7), padding=(0, 3), groups=self.width, bias=False),
            nn.Conv2d(self.width, self.width, kernel_size=(7, 1), padding=(3, 0), groups=self.width, bias=False),
            nn.BatchNorm2d(self.width))
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * self.width, self.width, kernel_size=(1, 1), groups=self.width, bias=False),
            nn.BatchNorm2d(self.width))

        self.conv2_0 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(3, 3), padding=(4, 4), dilation=(4, 4), bias=False),
            nn.BatchNorm2d(self.width))
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(1, 11), padding=(0, 5), groups=self.width, bias=False),
            nn.Conv2d(self.width, self.width, kernel_size=(11, 1), padding=(5, 0), groups=self.width, bias=False),
            nn.BatchNorm2d(self.width))
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * self.width, self.width, kernel_size=(1, 1), groups=self.width, bias=False),
            nn.BatchNorm2d(self.width))

        self.conv3_0 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(3, 3), padding=(8, 8), dilation=(8, 8), bias=False),
            nn.BatchNorm2d(self.width))
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(1, 21), padding=(0, 10), groups=self.width, bias=False),
            nn.Conv2d(self.width, self.width, kernel_size=(21, 1), padding=(10, 0), groups=self.width, bias=False),
            nn.BatchNorm2d(self.width))
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * self.width, self.width, kernel_size=(1, 1), groups=self.width, bias=False),
            nn.BatchNorm2d(self.width))

        self.channel_squeeze = nn.Sequential(
            nn.Conv2d(scale * self.width, dim, kernel_size=(1, 1), groups=dim, bias=False),
            nn.BatchNorm2d(dim))

    def forward(self, x):
        out = self.act(self.channel_expand(x))
        spx = torch.split(out, self.width, 1)

        out0 = self.act(self.conv0(self.act(torch.cat((self.conv0_0(spx[0]), self.conv0_1(spx[0])), dim=1))))
        out1 = self.act(self.conv1(self.act(torch.cat((self.conv1_0(spx[1]), self.conv1_1(spx[1])), dim=1))))
        out2 = self.act(self.conv2(self.act(torch.cat((self.conv2_0(spx[2]), self.conv2_1(spx[2])), dim=1))))
        out3 = self.act(self.conv3(self.act(torch.cat((self.conv3_0(spx[3]), self.conv3_1(spx[3])), dim=1))))

        out = torch.cat((out0, out1, out2, out3), dim=1)
        out = self.act(self.act(self.channel_squeeze(out)) + x)

        return out


class ReceptiveConv_1(nn.Module):
    def __init__(self, dim, expand=2, scale=4):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(ReceptiveConv_1, self).__init__()
        assert scale >= 1, 'The input scale must be a positive value'

        self.width = int(dim * 2 / scale)
        self.relu = nn.ReLU(inplace=True)
        self.channel_expand = nn.Sequential(
            nn.Conv2d(dim, dim * expand, kernel_size=(1, 1), groups=dim, bias=False),
            nn.BatchNorm2d(dim * expand))

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(3, 3), padding=(1, 1),
                      dilation=(1, 1), groups=self.width, bias=False),
            nn.BatchNorm2d(self.width))

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(3, 3), padding=(2, 2),
                      dilation=(2, 2), groups=self.width, bias=False),
            nn.BatchNorm2d(self.width))

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(3, 3), padding=(4, 4),
                      dilation=(4, 4), groups=self.width, bias=False),
            nn.BatchNorm2d(self.width))

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(3, 3), padding=(8, 8),
                      dilation=(8, 8), groups=self.width, bias=False),
            nn.BatchNorm2d(self.width))

        self.channel_squeeze = nn.Sequential(
            nn.Conv2d(scale * self.width, dim, kernel_size=(1, 1), groups=dim, bias=False),
            nn.BatchNorm2d(dim))

    def forward(self, x):
        out = self.relu(self.channel_expand(x))

        spx = torch.split(out, self.width, 1)
        out0 = self.relu(self.conv0(spx[0]))
        out1 = self.relu(self.conv1(spx[1] + spx[0]))
        out2 = self.relu(self.conv2(spx[2] + spx[1] + spx[0]))
        out3 = self.relu(self.conv3(spx[3] + spx[2] + spx[1] + spx[0]))

        out = torch.cat((out0, out1, out2, out3), dim=1)
        out = self.channel_squeeze(out)
        out = self.relu(out + x)

        return out + x


# class AttentionModule_1(nn.Module):
#     def __init__(self, dim, expand=2, scale=4):
#         super(AttentionModule_1, self).__init__()
#         self.width = int(dim * expand / scale)
#         self.channel_expand = nn.Sequential(
#             nn.Conv2d(dim, dim * expand, kernel_size=(1, 1), groups=dim),
#             nn.BatchNorm2d(dim * expand))
#
#         self.conv0 = nn.Sequential(
#             nn.Conv2d(self.width, self.width, kernel_size=(1, 5), padding=(0, 2), groups=self.width),
#             nn.Conv2d(self.width, self.width, kernel_size=(5, 1), padding=(2, 0), groups=self.width),
#             nn.BatchNorm2d(self.width))
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(self.width, self.width, kernel_size=(1, 7), padding=(0, 3), groups=self.width),
#             nn.Conv2d(self.width, self.width, kernel_size=(7, 1), padding=(3, 0), groups=self.width),
#             nn.BatchNorm2d(self.width))
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(self.width, self.width, kernel_size=(1, 11), padding=(0, 5), groups=self.width),
#             nn.Conv2d(self.width, self.width, kernel_size=(11, 1), padding=(5, 0), groups=self.width),
#             nn.BatchNorm2d(self.width))
#
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(self.width, self.width, kernel_size=(1, 21), padding=(0, 10), groups=self.width),
#             nn.Conv2d(self.width, self.width, kernel_size=(21, 1), padding=(10, 0), groups=self.width),
#             nn.BatchNorm2d(self.width))
#
#         self.channel_squeeze = nn.Sequential(
#             nn.Conv2d(self.width * scale, dim, kernel_size=(1, 1), groups=dim),
#             nn.BatchNorm2d(dim))
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.relu(self.channel_expand(x))
#         spx = torch.split(out, self.width, 1)
#
#         out0 = self.relu(self.conv0(spx[0]))
#         out1 = self.relu(self.conv1(spx[1]))
#         out2 = self.relu(self.conv2(spx[2]))
#         out3 = self.relu(self.conv3(spx[3]))
#
#         out = torch.cat((out0, out1, out2, out3), dim=1)
#         out = self.relu(self.channel_squeeze(out)) + x
#
#         return out


# class AttentionModule(nn.Module):
#     def __init__(self, dim):
#         super(AttentionModule, self).__init__()
#         self.channel_expand = nn.Conv2d(dim, dim * 2, (1, 1), groups=dim)
#
#         self.conv0 = nn.Conv2d(dim, dim, (5, 5), (2, 2), groups=dim)
#         self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
#         self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
#         self.bn0 = nn.BatchNorm2d(dim)
#
#         self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
#         self.bn1 = nn.BatchNorm2d(dim)
#
#         self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
#         self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
#         self.bn2 = nn.BatchNorm2d(dim)
#
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#         self.bn3 = nn.BatchNorm2d(dim)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0(x)
#
#         attn_0 = self.conv0_1(attn)
#         attn_0 = self.relu(self.bn0(self.conv0_2(attn_0)))
#
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.relu(self.bn1(self.conv1_2(attn_1)))
#
#         attn_2 = self.conv2_1(attn)
#         attn_2 = self.relu(self.bn2(self.conv2_2(attn_2)))
#         attn = attn + attn_0 + attn_1 + attn_2
#
#         attn = self.relu(self.bn3(self.conv3(attn)))
#
#         return attn * u


# class MultiOrderDWConv(nn.Module):
#     """Multi-order Features with Dilated DWConv Kernel.
#     Args:
#         embed_dims (int): Number of input channels.
#         dw_dilation (list): Dilations of three DWConv layers.
#         channel_split (list): The raletive ratio of three splited channels.
#     """
#
#     def __init__(self,
#                  embed_dims,
#                  dw_dilation=None,
#                  channel_split=None,
#                  ):
#         super(MultiOrderDWConv, self).__init__()
#
#         if channel_split is None:
#             channel_split = [1, 3, 4, ]
#         if dw_dilation is None:
#             dw_dilation = [1, 2, 3, ]
#         self.split_ratio = [i / sum(channel_split) for i in channel_split]
#         self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
#         self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
#         self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
#         self.embed_dims = embed_dims
#         assert len(dw_dilation) == len(channel_split) == 3
#         assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
#         assert embed_dims % sum(channel_split) == 0
#
#         # basic DW conv
#         self.DW_conv0 = nn.Conv2d(
#             in_channels=self.embed_dims,
#             out_channels=self.embed_dims,
#             kernel_size=5,
#             padding=(1 + 4 * dw_dilation[0]) // 2,
#             groups=self.embed_dims,
#             stride=1, dilation=dw_dilation[0],
#         )
#         # DW conv 1
#         self.DW_conv1 = nn.Conv2d(
#             in_channels=self.embed_dims_1,
#             out_channels=self.embed_dims_1,
#             kernel_size=5,
#             padding=(1 + 4 * dw_dilation[1]) // 2,
#             groups=self.embed_dims_1,
#             stride=1, dilation=dw_dilation[1],
#         )
#         # DW conv 2
#         self.DW_conv2 = nn.Conv2d(
#             in_channels=self.embed_dims_2,
#             out_channels=self.embed_dims_2,
#             kernel_size=7,
#             padding=(1 + 6 * dw_dilation[2]) // 2,
#             groups=self.embed_dims_2,
#             stride=1, dilation=dw_dilation[2],
#         )
#         # a channel convolution
#         self.PW_conv = nn.Conv2d(  # point-wise convolution
#             in_channels=embed_dims,
#             out_channels=embed_dims,
#             kernel_size=1)
#
#     def forward(self, x):
#         x_0 = self.DW_conv0(x)
#         x_1 = self.DW_conv1(
#             x_0[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...])
#         x_2 = self.DW_conv2(
#             x_0[:, self.embed_dims - self.embed_dims_2:, ...])
#         x = torch.cat([
#             x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
#         x = self.PW_conv(x)
#         return x


class GCB(nn.Module):
    def __init__(self, dim=256):
        super(GCB, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.pool4 = nn.AdaptiveAvgPool2d((6, 6))

        self.conv0 = nn.Conv2d(2048, dim // 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv1 = nn.Conv2d(2048, dim // 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv2 = nn.Conv2d(2048, dim // 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv3 = nn.Conv2d(2048, dim // 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv4 = nn.Conv2d(2048, dim // 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.bn = nn.BatchNorm2d(dim // 4, eps=1e-5, momentum=0.01, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        shape = x.shape[2:]
        x_t = self.act(self.bn(self.conv1(self.pool1(x))))
        out1 = F.interpolate(self.act(self.bn(self.conv1(self.pool1(x)))),
                             shape, mode='bilinear', align_corners=False) * x_t
        out2 = F.interpolate(self.act(self.bn(self.conv2(self.pool2(x)))),
                             shape, mode='bilinear', align_corners=False) * x_t
        out3 = F.interpolate(self.act(self.bn(self.conv3(self.pool3(x)))),
                             shape, mode='bilinear', align_corners=False) * x_t
        out4 = F.interpolate(self.act(self.bn(self.conv4(self.pool4(x)))),
                             shape, mode='bilinear', align_corners=False) * x_t

        return torch.cat((out1, out2, out3, out4), dim=1)

# class BasicConv(nn.Module):
#
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=(0, 0), dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=(stride, stride), padding=padding,
#                               dilation=(dilation, dilation), groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU(inplace=True) if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
#
# class BasicRFB_a(nn.Module):
#
#     def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
#         super(BasicRFB_a, self).__init__()
#         self.scale = scale
#         self.out_channels = out_planes
#         inter_planes = in_planes // 4
#
#         self.branch0 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
#             BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
#             BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
#             BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
#         )
#         self.branch3 = nn.Sequential(
#             BasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
#             BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
#             BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
#             BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
#         )
#
#         self.ConvLinear = BasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
#         self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         x3 = self.branch3(x)
#
#         out = torch.cat((x0, x1, x2, x3), 1)
#         out = self.ConvLinear(out)
#         short = self.shortcut(x)
#         out = out * self.scale + short
#         out = self.relu(out)
#
#         return out


# class ResidualConvBlock(nn.Module):
#     def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
#                  bias=True, use_relu=True, use_bn=True, frozen=False):
#         super(ResidualConvBlock, self).__init__()
#         self.conv = ConvBNReLU(nIn, nOut, ksize=ksize, stride=stride, pad=pad,
#                                dilation=dilation, groups=groups, bias=bias,
#                                use_relu=use_relu, use_bn=use_bn, frozen=frozen)
#         self.residual_conv = ConvBNReLU(nIn, nOut, ksize=1, stride=stride, pad=0,
#                                         dilation=1, groups=groups, bias=bias,
#                                         use_relu=False, use_bn=use_bn, frozen=frozen)
#
#     def forward(self, x):
#         x = self.conv(x) + self.residual_conv(x)
#         return x


# class PPM(nn.Module):
#     def __init__(self):
#         super(PPM, self).__init__()
#         self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
#         self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
#         self.pool3 = nn.AdaptiveAvgPool2d((3, 3))
#         self.pool4 = nn.AdaptiveAvgPool2d((6, 6))
#
#         self.conv1 = nn.Conv2d(2048, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
#         self.conv2 = nn.Conv2d(2048, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
#         self.conv3 = nn.Conv2d(2048, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
#         self.conv4 = nn.Conv2d(2048, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
#
#         self.conv5 = nn.Conv2d(4, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
#
#     def forward(self, x, shape):
#         out = F.interpolate(self.conv1(self.pool1(x)), shape, mode='bilinear', align_corners=False)
#         out = torch.cat((out, F.interpolate(self.conv2(self.pool2(x)), shape, mode='bilinear', align_corners=False)),
#                         dim=1)
#         out = torch.cat((out, F.interpolate(self.conv3(self.pool3(x)), shape, mode='bilinear', align_corners=False)),
#                         dim=1)
#         out = torch.cat((out, F.interpolate(self.conv4(self.pool4(x)), shape, mode='bilinear', align_corners=False)),
#                         dim=1)
#         out = self.conv5(out)
#
#         return out


# class ASPPConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, dilation):
#         modules = [
#             nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         ]
#         super(ASPPConv, self).__init__(*modules)


# class ASPPPooling(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super(ASPPPooling, self).__init__(
#             nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU())
#
#     def forward(self, x):
#         size = x.shape[-2:]
#         for mod in self:
#             x = mod(x)
#         # 上采样
#         return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
#
#     # 整个 ASPP 架构


# class ASPP(nn.Module):
#     def __init__(self, in_channels, atrous_rates, out_channels=256):
#         super(ASPP, self).__init__()
#         modules = []
#         # 1*1 卷积
#         modules.append(nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()))
#
#         # 多尺度空洞卷积
#         rates = tuple(atrous_rates)
#         for rate in rates:
#             modules.append(ASPPConv(in_channels, out_channels, rate))
#
#         # 池化
#         modules.append(ASPPPooling(in_channels, out_channels))
#
#         self.convs = nn.ModuleList(modules)
#
#         # 拼接后的卷积
#         self.project = nn.Sequential(
#             nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Dropout(0.5))
#
#     def forward(self, x):
#         res = []
#         for conv in self.convs:
#             res.append(conv(x))
#         res = torch.cat(res, dim=1)
#         return self.project(res)
