import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.utils import get_root_logger
from ..builder import BACKBONES

class BasicBlock(nn.Module):
    """Basic block for ResNet."""

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    """for unet"""

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


class DownSample_conv(nn.Module):
    """Downscaling with maxpool then double conv for unet"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 ):
        super(DownSample_conv, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels=in_channels,
                       out_channels=out_channels,
                       conv_cfg=self.conv_cfg,
                       norm_cfg=self.norm_cfg,
                       act_cfg=self.act_cfg,
                       )
        )

    def forward(self, x):
        return self.maxpool_conv(x)


@BACKBONES.register_module()
class UNet(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_indices=(0, 1, 2, 3),
                 u_net_channels=(16, 32, 64, 128),
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 zero_init_residual=True):
        super(UNet, self).__init__()
        self.out_indices = out_indices
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        # self.shuffle_factor = shuffle_factor,
        self.zero_init_residual = zero_init_residual
        self.unet_channels = u_net_channels
        '''need complete'''
        # self.make_res_layer()
        self.res_layers = []
        self.unet_layers = []

        self.inc_u = DoubleConv(in_channels=in_channels,
                                out_channels=self.unet_channels[0],
                                conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg,
                                )

        for i in range(0, len(u_net_channels) - 1):
            unet_layer = DownSample_conv(in_channels=self.unet_channels[i],
                                         out_channels=self.unet_channels[i + 1],
                                         conv_cfg=self.conv_cfg,
                                         norm_cfg=self.norm_cfg,
                                         )
            layer_name = f'u_layer{i + 1}'
            self.add_module(layer_name, unet_layer)
            self.unet_layers.append(layer_name)
        '''need complete'''

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

    def forward(self, x):
        """Forward function."""
        """U part"""
        y = self.inc_u(x)
        out_u = []
        out_u.append(y)
        for i, layer_name in enumerate(self.unet_layers):
            u_net_layer = getattr(self, layer_name)
            y = u_net_layer(y)
            if i in self.out_indices:
                out_u.append(y)
        return tuple(out_u)
