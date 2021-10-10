import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmseg.ops import resize, Upsample
from ..builder import NECKS
import torch

'''class LeakyUnit(nn.Module):
    def __init__(self, n_features):
        super(LeakyUnit, self).__init__()
        self.W_r = nn.Conv2d(2 * n_features, n_features, kernel_size=3, padding=1, stride=1, bias=False)
        self.W = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1, stride=1, bias=False)
        self.U = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1, stride=1, bias=False)
        self.W_z = nn.Conv2d(2 * n_features, n_features, kernel_size=3, padding=1, stride=1, bias=False)
        self.sigma = nn.Sigmoid()

    def forward(self, f_m, f_n):
        r_mn = self.sigma(self.W_r(torch.cat((f_m, f_n), dim=1)))
        f_mn_hat = torch.tanh(self.U(f_m) + self.W(r_mn * f_n))
        z_mn = self.sigma(self.W_z(torch.cat((f_m, f_n), dim=1)))
        f_m_out = z_mn * f_m + (1 - z_mn) * f_mn_hat
        # f_n_out = (1 - r_mn) * f_n

        return f_m_out, r_mn, z_mn'''


class LeakyUnit(nn.Module):
    def __init__(self, n_fi, n_fo):
        super(LeakyUnit, self).__init__()
        self.W_r = nn.Conv2d(n_fi + n_fo, n_fo, kernel_size=3, padding=1, stride=1, bias=False)
        self.W = nn.Conv2d(n_fo, n_fo, kernel_size=3, padding=1, stride=1, bias=False)
        self.U = nn.Conv2d(n_fi, n_fo, kernel_size=3, padding=1, stride=1, bias=False)
        self.W_z = nn.Conv2d(n_fi + n_fo, n_fo, kernel_size=3, padding=1, stride=1, bias=False)
        self.sigma = nn.Sigmoid()

    def forward(self, f_m, f_n):
        r_mn = self.sigma(self.W_r(torch.cat((f_m, f_n), dim=1)))
        f_mn_hat = torch.tanh(self.U(f_m) + self.W(r_mn * f_n))
        z_mn = self.sigma(self.W_z(torch.cat((f_m, f_n), dim=1)))
        f_m_out = z_mn * f_m + (1 - z_mn) * f_mn_hat
        # f_n_out = (1 - r_mn) * f_n

        return f_m_out, r_mn, z_mn


class LeakyUnit_adaptive(nn.Module):
    def __init__(self, n_fi, n_fo, co_ch, conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super(LeakyUnit_adaptive, self).__init__()
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        # self.co_ch = co_ch
        self.fit_m = ConvModule(n_fo, co_ch, kernel_size=1, padding=0,
                                stride=1, bias=False,
                                conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg
                                )
        self.fit_n = ConvModule(n_fi, co_ch, kernel_size=1, padding=0,
                                stride=1, bias=False,
                                conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg
                                )
        self.W_r = ConvModule(co_ch + co_ch, co_ch, kernel_size=1, padding=0,
                              stride=1, bias=False,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg,
                              act_cfg=self.act_cfg
                              )
        self.W = ConvModule(co_ch, co_ch, kernel_size=1, padding=0,
                            stride=1, bias=False,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg
                            )
        self.U = ConvModule(co_ch, co_ch, kernel_size=1, padding=0,
                            stride=1, bias=False,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg
                            )
        self.W_z = ConvModule(co_ch + co_ch, co_ch, kernel_size=1, padding=0,
                              stride=1, bias=False,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg,
                              act_cfg=self.act_cfg
                              )
        self.sigma = nn.Sigmoid()
        self.out = ConvModule(co_ch, n_fo, kernel_size=1, padding=0,
                              stride=1, bias=False,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg,
                              act_cfg=self.act_cfg
                              )

        '''ConvModule(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)'''

    def forward(self, f_m, f_n):
        # prev_shape = laterals[i - 1].shape[2:]
        # laterals[i - 1] += F.interpolate(
        #     laterals[i], size=prev_shape, **self.upsample_cfg)
        prev_shape = f_m.shape[2:]
        f_n = F.interpolate(f_n, size=prev_shape)
        f_m = self.fit_m(f_m)
        f_n = self.fit_n(f_n)
        r_mn = self.sigma(self.W_r(torch.cat((f_m, f_n), dim=1)))
        f_mn_hat = torch.tanh(self.U(f_m) + self.W(r_mn * f_n))
        z_mn = self.sigma(self.W_z(torch.cat((f_m, f_n), dim=1)))
        f_m_out = z_mn * f_m + (1 - z_mn) * f_mn_hat
        # f_n_out = (1 - r_mn) * f_n
        f_m_out = self.out(f_m_out)
        return f_m_out, r_mn, z_mn

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg

        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            mid_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpSampleFeatureFusionModel(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 bilinear=True,
                 align_corners=False
                 ):
        super(UpSampleFeatureFusionModel, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners)  # 'nearest'/bilinear
            # self.conv1 = FusionLayer(in_channels,in_channels)
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # self.conv1 = FusionLayer(in_channels, in_channels)
            self.conv = ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, higher_features, lower_features):
        higher_features = self.up(higher_features)
        # input is CHW
        diffY = torch.tensor([lower_features.size()[2] - higher_features.size()[2]])
        diffX = torch.tensor([lower_features.size()[3] - higher_features.size()[3]])
        higher_features = F.pad(higher_features, [diffX // 2, diffX - diffX // 2,
                                                  diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        fused_features = torch.cat([lower_features, higher_features], dim=1)
        # x = self.conv1(x)
        return self.conv(fused_features)


@NECKS.register_module()
class FPN_U(nn.Module):
    """Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN_U(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 inu_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN_U, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.inu_channels = inu_channels
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.leakyunit_u_fpn = nn.ModuleList()
        self.leakyunit_fpn_u = nn.ModuleList()
        self.adaptive_channel = 256
        # self.LeakyUnit_adaptive=LeakyUnit_adaptive
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'
        '''u part'''
        self.u_convs = nn.ModuleList()
        self.lateral_u = nn.ModuleList()
        for i in range(0, len(inu_channels) - 1):
            up_conv = UpSampleFeatureFusionModel(in_channels=self.inu_channels[i] + self.inu_channels[i + 1],
                                                 out_channels=self.inu_channels[i],
                                                 conv_cfg=conv_cfg,
                                                 norm_cfg=norm_cfg,
                                                 )
            u_conv = ConvModule(
                self.inu_channels[i],
                self.inu_channels[i],
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_u.append(up_conv)
            self.u_convs.append(u_conv)

        '''
        only the last 3 layers of FPN_seg and the first 3 layers of UNet_gan implement soft selection
        '''
        for i in range(0, len(inu_channels)-1):

            self.leakyunit_u_fpn.append(LeakyUnit_adaptive(self.inu_channels[i+1],
                                                           self.out_channels, self.adaptive_channel))
            self.leakyunit_fpn_u.append(LeakyUnit_adaptive(self.out_channels,
                                                           self.inu_channels[i+1], self.adaptive_channel))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs_all):
        assert len(inputs_all[0]) == len(self.in_channels)
        inputs = inputs_all[0]
        inputs_u = inputs_all[1]
        # laterals_u = []
        '''u_part'''
        # for i, lateral_uconv in enumerate(self.lateral_u):
        #     laterals_u = lateral_uconv(inputs_u)
        """build top to down u net"""
        # u_ori = inputs_u[-1]

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        u_feat = inputs_u[-1]
        # firstly implement feature selection for the top feature:
        # u_feat, _, _ = self.leakyunit_fpn_u[-1](inputs_u[-1], laterals[-1])
        # laterals[-1], _, _ = self.leakyunit_u_fpn[-1](laterals[-1], inputs_u[-1])
        prev_shape = laterals[-2].shape[2:]
        laterals[-2] += F.interpolate(
            laterals[-1], size=prev_shape, **self.upsample_cfg)


        for i in range(used_backbone_levels - 2, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            '''
            u_feature & lateral_fpn_feature: soft_selection -> u_feat_hat, lateral_fpn
            u_feat_hat with low_layer_u_input: concatenation & conv -> u_feat 
            '''
            # u_feat_hat, _, _ = self.leakyunit_fpn_u[i](u_feat, laterals[i])
            # laterals[i], _, _ = self.leakyunit_u_fpn[i](laterals[i], u_feat)
            lateral_u = self.lateral_u[i](u_feat, inputs_u[i]) # implement up_sampling
            # laterals_u.append(lateral_u)
            u_feat = self.u_convs[i](lateral_u)

            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
        # last_layer
        # u_feat_hat, _, _ = self.leakyunit_fpn_u[0](u_feat, laterals[0])
        # laterals[0], _, _ = self.leakyunit_u_fpn[0](laterals[0], u_feat)
        lateral_u = self.lateral_u[0](u_feat, inputs_u[0])
        # lateral_u_hat, _, _ = self.leakyunit_fpn_u[0](laterals[0], lateral_u)
        # laterals[0], _, _ = self.leakyunit_u_fpn[0](lateral_u, laterals[0])
        out_u = self.u_convs[0](lateral_u)
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs), out_u
