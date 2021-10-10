from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from mmcv.cnn import normal_init, ConvModule, kaiming_init
from mmseg.models.builder import HEADS
from mmseg.ops import resize, Upsample
import torch.nn.functional as F
from .decode_head import BaseDecodeHead


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
                 bilinear=False,
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
            self.conv = DoubleConv(
                out_channels,
                out_channels,
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
        '''fused_features = torch.cat([lower_features, higher_features], dim=1)'''
        # x = self.conv1(x)
        fused_features = higher_features + lower_features
        return self.conv(fused_features)


@HEADS.register_module()
class BaseGHead(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int): The label index to be ignored. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(mode='nearest'),
                 conv_module=ConvModule):
        super(BaseGHead, self).__init__()

        self.in_channels = in_channels
        self.decode_convs = nn.ModuleList()
        for i in range(0, len(in_channels) - 1):
            up_conv = UpSampleFeatureFusionModel(in_channels=self.in_channels[i+1],
                                                 out_channels=self.in_channels[i],
                                                 conv_cfg=conv_cfg,
                                                 norm_cfg=norm_cfg,
                                                 act_cfg=act_cfg,
                                                 )
            self.decode_convs.append(up_conv)

        self.out = conv_module(
            in_channels[0],
            out_channels,
            1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))

    def forward(self, x, x_in):
        """Forward function.

        Args:
            x_in:
            x (Tensor): Input feature map with shape (N, C, H, W).
            shortcut (Tensor): The shorcut connection with shape
                (N, C, H', W').
            dec_idx_feat (Tensor, optional): The decode index feature map with
                shape (N, C, H', W'). Defaults to None.

        Returns:
            Tensor: Output tensor with shape (N, C, H', W').
        """
        u_feat = x[-1]
        for i in range(len(self.in_channels) - 1, 0, -1):
            u_feat = self.decode_convs[i-1](u_feat, x[i-1])  # implement up_sampling
        return self.out(u_feat) + x_in

@HEADS.register_module()
class UNetHead(BaseDecodeHead):

    def __init__(self,
                 conv_module=ConvModule, **kwargs):
        super(UNetHead, self).__init__(**kwargs)
        self.decode_convs = nn.ModuleList()
        for i in range(0, len(self.in_channels) - 1):
            up_conv = UpSampleFeatureFusionModel(in_channels=self.in_channels[i+1],
                                                 out_channels=self.in_channels[i],
                                                 conv_cfg=self.conv_cfg,
                                                 norm_cfg=self.norm_cfg,
                                                 act_cfg=self.act_cfg,
                                                 )
            self.decode_convs.append(up_conv)

        self.out = nn.Conv2d(
            self.in_channels[0],
            self.num_classes,
            1)

    def forward(self, x):
        """Forward function.

        Args:
            x_in:
            x (Tensor): Input feature map with shape (N, C, H, W).
            shortcut (Tensor): The shorcut connection with shape
                (N, C, H', W').
            dec_idx_feat (Tensor, optional): The decode index feature map with
                shape (N, C, H', W'). Defaults to None.

        Returns:
            Tensor: Output tensor with shape (N, C, H', W').
        """
        u_feat = x[-1]
        for i in range(len(self.in_channels) - 1, 0, -1):
            u_feat = self.decode_convs[i-1](u_feat, x[i-1])  # implement up_sampling

        return self.out(u_feat)



