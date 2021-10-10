import torch
import torch.nn as nn
from mmcv.cnn import normal_init, ConvModule, kaiming_init
from mmcv.runner import auto_fp16, force_fp32
from mmseg.models.builder import GENERATOR_HEAD
from abc import ABCMeta, abstractmethod

@GENERATOR_HEAD.register_module()
class SingleGHead(nn.Module, metaclass=ABCMeta):
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
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 conv_module=ConvModule):
        super(SingleGHead, self).__init__()

        self.conv = conv_module(
            in_channels,
            in_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.out = conv_module(
            in_channels,
            out_channels,
            1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))

    #     self.init_weights()
    #
    # def init_weights(self):
    #     """Init weights for the module.
    #     """
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             kaiming_init(m, mode='fan_in', nonlinearity='leaky_relu')

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
        x = self.conv(x)
        return self.out(x) + x_in
