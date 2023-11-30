import torch
from torch import nn
import torch.nn.functional as F
from net.resnet import Bottleneck, ResNet, FrozenBatchNorm2d
from net.modules import CBR, ReceptiveConv, ReceptiveConv_1, Object_Refinement_module, Fourier_Edge_extractor, GCB
from net.EfficientNet import EfficientNet
from net.burger import get_hamburger
from net import settings


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class OrNet(nn.Module):
    def __init__(self, cfg, pretrained=None, use_carafe=True,
                 enc_channels=None,
                 dec_channels=None, freeze_s1=False):
        super(OrNet, self).__init__()

        if 'resnet50' in cfg.model_choose:
            enc_channels = [64, 256, 512, 1024, 2048, 1024, 1024]
            dec_channels = [32, 64, 256, 512, 512, 128, 128]
            self.inplanes = enc_channels[-3]
            self.base_width = 64
            self.encoder = ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=FrozenBatchNorm2d)
        elif 'efficientnet' in cfg.model_choose:
            if cfg.version == '1':
                enc_channels = [32, 24, 40, 112, 320, 112, 112]
                dec_channels = [16, 12, 24, 48, 64, 48, 48]
            elif cfg.version == '2':
                enc_channels = [32, 24, 48, 120, 352, 120, 120]
                dec_channels = [16, 12, 24, 48, 64, 48, 48]
            elif cfg.version == '3':
                enc_channels = [40, 32, 48, 136, 384, 136, 136]
                dec_channels = [16, 12, 24, 48, 64, 48, 48]
            elif cfg.version == '4':
                enc_channels = [48, 32, 56, 160, 448, 160, 160]
                dec_channels = [16, 12, 24, 48, 64, 48, 48]
            elif cfg.version == '5':
                enc_channels = [48, 40, 64, 176, 512, 176, 176]
                dec_channels = [24, 20, 32, 64, 128, 64, 64]
            elif cfg.version == '6':
                enc_channels = [56, 40, 72, 200, 576, 200, 200]
                dec_channels = [24, 20, 32, 64, 128, 64, 64]
            elif cfg.version == '7':
                enc_channels = [64, 48, 80, 224, 640]
                dec_channels = [24, 20, 32, 64, 128, 64, 64]
            elif cfg.version == '8':
                enc_channels = [32, 24, 48, 120, 352, 120, 120]
                dec_channels = [24, 20, 32, 64, 128, 64, 64]

            self.inplanes = enc_channels[-3]
            self.base_width = 64
            self.encoder = EfficientNet.from_pretrained(f'efficientnet-b{cfg.version}', advprop=True)

        self.FG_branch = cfg.FG_branch
        self.BG_branch = cfg.BG_branch
        self.decoder_fg = Decoder_FG(enc_channels, dec_channels)
        self.decoder_bg = Decoder_BG(enc_channels, dec_channels)
        self.cls_fg = nn.Sequential(nn.Conv2d(dec_channels[0], 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                    nn.Conv2d(dec_channels[1], 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                    nn.Conv2d(dec_channels[2], 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                    nn.Conv2d(dec_channels[3], 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                    nn.Conv2d(dec_channels[4], 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        self.cls_bg = nn.Sequential(nn.Conv2d(dec_channels[0], 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                    nn.Conv2d(dec_channels[1], 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                    nn.Conv2d(dec_channels[2], 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                    nn.Conv2d(dec_channels[3], 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                    nn.Conv2d(dec_channels[4], 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv6 = nn.Sequential(
            self._make_layer(enc_channels[-2] // 4, 2, stride=2),
        )
        self.conv7 = nn.Sequential(
            self._make_layer(enc_channels[-2] // 4, 2, stride=2),
        )
        self.fee_fg = Fourier_Edge_extractor(radius=cfg.frequency_radius, channel=dec_channels[1])
        self.fee_bg = Fourier_Edge_extractor(radius=cfg.frequency_radius, channel=dec_channels[1])

        # self.RFB_2 = BasicRFB_a(512, 512)
        # self.RFB_3 = BasicRFB_a(1024, 1024)
        # self.RFB_4 = BasicRFB_a(2048, 2048)

        # self.hamburger_fg = get_hamburger(settings.VERSION)(settings.CHANNELS, settings)
        # self.ham_in_channel_squeeze_fg = nn.Conv2d(dec_channels[-1] + dec_channels[-2] + dec_channels[-3],
        #                                            512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # self.ham_squeeze_fg = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        #
        # self.hamburger_bg = get_hamburger(settings.VERSION)(settings.CHANNELS, settings)
        # self.ham_in_channel_squeeze_bg = nn.Conv2d(dec_channels[-1] + dec_channels[-2] + dec_channels[-3],
        #                                            512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # self.ham_squeeze_bg = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = 1
        groups = 1
        expansion = 4
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * expansion, stride),
                norm_layer(planes * expansion),
            )

        layers = [Bottleneck(self.inplanes, planes, stride, downsample, groups,
                             self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=groups,
                                     base_width=self.base_width, dilation=1,
                                     norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, input):
        x0, x1, x2, x3, x4 = self.encoder(input)

        x5 = self.conv6(x4)
        x6 = self.conv7(x5)
        att = torch.sigmoid(self.gap(x6))

        saliency_maps_fg = []
        saliency_maps_bg = []

        if self.FG_branch:
            f_dec_fg = self.decoder_fg([x0, x1, x2, x3, x4, x5, x6], att)
            for idx, feature in enumerate(f_dec_fg[0:5]):
                out_map_fg = self.cls_fg[idx](feature)
                saliency_maps_fg.append(F.interpolate(out_map_fg, input.shape[2:], mode='bilinear', align_corners=False))
            saliency_maps_fg = torch.sigmoid(torch.cat(saliency_maps_fg, dim=1))

            # ----------FEE module----------
            fee_edge_fg = self.fee_fg(f_dec_fg[1])
            fee_edge_fg = F.interpolate(fee_edge_fg, input.shape[2:], mode='bilinear', align_corners=False)
            saliency_maps_fg = torch.cat((saliency_maps_fg, torch.sigmoid(fee_edge_fg)), dim=1)
            # ----------FEE module----------

            # # ----------Hamburger module----------
            # ham_in_fg = torch.cat((f_dec_fg[2],
            #                        F.interpolate(f_dec_fg[3], f_dec_fg[2].shape[2:], mode='bilinear',
            #                                      align_corners=False),
            #                        F.interpolate(f_dec_fg[4], f_dec_fg[2].shape[2:], mode='bilinear',
            #                                      align_corners=False)),
            #                       dim=1)
            # ham_in_fg = self.ham_in_channel_squeeze_fg(ham_in_fg)
            # ham_out_fg = self.hamburger_fg(ham_in_fg)
            # ham_out_fg = F.interpolate(self.ham_squeeze_fg(ham_out_fg), input.shape[2:], mode='bilinear',
            #                            align_corners=False)
            # saliency_maps_fg = torch.cat((saliency_maps_fg, torch.sigmoid(ham_out_fg)), dim=1)
            # # ----------Hamburger module----------

        if self.BG_branch:
            f_dec_bg = self.decoder_bg([x0, x1, x2, x3, x4, x5, x6], att)
            for idx, feature in enumerate(f_dec_bg[0:5]):
                out_map_bg = self.cls_bg[idx](feature)
                saliency_maps_bg.append(F.interpolate(out_map_bg, input.shape[2:], mode='bilinear', align_corners=False))
            saliency_maps_bg = torch.sigmoid(torch.cat(saliency_maps_bg, dim=1))

            # ----------FEE module----------
            fee_edge_bg = self.fee_bg(f_dec_bg[1])
            fee_edge_bg = F.interpolate(fee_edge_bg, input.shape[2:], mode='bilinear', align_corners=False)
            saliency_maps_bg = torch.cat((saliency_maps_bg, torch.sigmoid(fee_edge_bg)), dim=1)
            # ----------FEE module----------

            # # ----------Hamburger module----------
            # ham_in_bg = torch.cat((f_dec_bg[2],
            #                        F.interpolate(f_dec_bg[3], f_dec_bg[2].shape[2:], mode='bilinear', align_corners=False),
            #                        F.interpolate(f_dec_bg[4], f_dec_bg[2].shape[2:], mode='bilinear', align_corners=False)), dim=1)
            # ham_in_bg = self.ham_in_channel_squeeze_bg(ham_in_bg)
            # ham_out_bg = self.hamburger_bg(ham_in_bg)
            # ham_out_bg = F.interpolate(self.ham_squeeze_bg(ham_out_bg), input.shape[2:], mode='bilinear', align_corners=False)
            # saliency_maps_bg = torch.cat((saliency_maps_bg, torch.sigmoid(ham_out_bg)), dim=1)
            # # ----------Hamburger module----------

        if self.FG_branch:
            if self.BG_branch:
                return saliency_maps_fg, 1 - saliency_maps_bg
            else:
                return saliency_maps_fg, saliency_maps_bg
        return saliency_maps_fg, 1 - saliency_maps_bg


class Decoder_FG(nn.Module):
    def __init__(self, in_channels, out_channels, use_dwconv=False):
        super(Decoder_FG, self).__init__()
        self.ca = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.ca.append(CBR(in_channels[i], out_channels[i] // 2, ksize=1, pad=0))
            self.up.append(CBR(out_channels[i + 1], out_channels[i] // 2, ksize=1, pad=0))
        self.ca.append(CBR(in_channels[-1], out_channels[-1], ksize=1, pad=0))
        self.gcb = GCB(dim=out_channels[-1] // 2)
        baseWidth = [32] * (len(in_channels) - 5) + [24] * 5

        self.fuse = nn.ModuleList()
        print("using dwconv:", use_dwconv)
        for i in range(len(in_channels)):
            # branch1 = ReceptiveConv(out_channels[i], out_channels[i], baseWidth=baseWidth[i],
            #                         dilation=dilation[i], use_dwconv=use_dwconv)

            # branch1 = MultiOrderDWConv(out_channels[i])
            # branch1 = ReceptiveConv(out_channels[i], out_channels[i], baseWidth=baseWidth[i],
            #                         dilation=[1, 2, 4, 8], use_dwconv=use_dwconv)
            # branch2 = ReceptiveConv(out_channels[i], out_channels[i], baseWidth=baseWidth[i],
            #                         dilation=[1, 2, 4, 8], use_dwconv=use_dwconv)

            branch1 = Object_Refinement_module(out_channels[i])
            branch2 = Object_Refinement_module(out_channels[i])

            # branch1 = ReceptiveConv_1(out_channels[i])
            # branch1 = AttentionModule(out_channels[i])

            self.fuse.append(nn.Sequential(branch1, branch2))

    def forward(self, features, att):
        # stage_result = self.fuse[-1](torch.cat(((self.ca[-1](features[-1])), self.gcb(features[-1])), dim=1))
        # stage_result = self.fuse[-1](torch.cat(((self.ca[-1](features[-1])), self.ca[-1](features[-1])), dim=1))

        if att is not None:
            stage_result = self.fuse[-1](self.ca[-1](features[-1] * att))
        else:
            stage_result = self.fuse[-1](self.ca[-1](features[-1]))
        results = [stage_result]
        num_mul_att = 1
        for idx in range(len(features) - 2, -1, -1):
            fea = self.up[idx](stage_result)
            inner_top_down = F.interpolate(fea, size=features[idx].shape[2:], mode='bilinear', align_corners=False)
            if att is not None and att.shape[1] == features[idx].shape[1] and num_mul_att:
                features[idx] = features[idx] * att
                num_mul_att -= 1
            inner_lateral = self.ca[idx](features[idx])
            stage_result = self.fuse[idx](torch.cat((inner_top_down, inner_lateral), dim=1))
            results.insert(0, stage_result)

        return results


class Decoder_BG(nn.Module):
    def __init__(self, in_channels, out_channels, use_dwconv=False):
        super(Decoder_BG, self).__init__()
        self.ca = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.ca.append(CBR(in_channels[i], out_channels[i] // 2, ksize=1, pad=0))
            self.up.append(CBR(out_channels[i + 1], out_channels[i] // 2, ksize=1, pad=0))
        self.ca.append(CBR(in_channels[-1], out_channels[-1], ksize=1, pad=0))
        self.gcb = GCB(dim=out_channels[-1] // 2)
        baseWidth = [32] * (len(in_channels) - 5) + [24] * 5

        self.fuse = nn.ModuleList()
        print("using dwconv:", use_dwconv)
        for i in range(len(in_channels)):
            # branch1 = ReceptiveConv(out_channels[i], out_channels[i], baseWidth=baseWidth[i],
            #                         dilation=dilation[i], use_dwconv=use_dwconv)

            # branch1 = MultiOrderDWConv(out_channels[i])
            # branch1 = ReceptiveConv(out_channels[i], out_channels[i], baseWidth=baseWidth[i],
            #                         dilation=[1, 2, 4, 8], use_dwconv=use_dwconv)
            # branch2 = ReceptiveConv(out_channels[i], out_channels[i], baseWidth=baseWidth[i],
            #                         dilation=[1, 2, 4, 8], use_dwconv=use_dwconv)

            branch1 = Object_Refinement_module(out_channels[i])
            branch2 = Object_Refinement_module(out_channels[i])

            # branch1 = ReceptiveConv_1(out_channels[i])
            # branch1 = AttentionModule(out_channels[i])

            self.fuse.append(nn.Sequential(branch1, branch2))

    def forward(self, features, att):
        # GCB Module
        # stage_result = self.fuse[-1](torch.cat(((self.ca[-1](features[-1])), self.gcb(features[-1])), dim=1))
        # stage_result = self.fuse[-1](torch.cat(((self.ca[-1](features[-1])), self.ca[-1](features[-1])), dim=1))

        if att is not None:
            stage_result = self.fuse[-1](self.ca[-1](features[-1] * att))
        else:
            stage_result = self.fuse[-1](self.ca[-1](features[-1]))
        results = [stage_result]
        num_mul_att = 1
        for idx in range(len(features) - 2, -1, -1):
            fea = self.up[idx](stage_result)
            inner_top_down = F.interpolate(fea, size=features[idx].shape[2:], mode='bilinear', align_corners=False)
            if att is not None and att.shape[1] == features[idx].shape[1] and num_mul_att:
                features[idx] = features[idx] * att
                num_mul_att -= 1
            inner_lateral = self.ca[idx](features[idx])
            stage_result = self.fuse[idx](torch.cat((inner_top_down, inner_lateral), dim=1))
            results.insert(0, stage_result)

        return results


def build_model(cfg):
    return OrNet(cfg)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
