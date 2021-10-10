import torch.nn as nn
from .. import builder
from ..builder import GENERATOR
import logging


@GENERATOR.register_module()
class base_gan(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 generator_head=None
                 ):
        super(base_gan, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
            self.with_neck = True
        else:
            self.with_neck = False

        self.G_head = builder.build_generator_head(generator_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.with_neck = None
        '''build discriminator'''
        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))
        if self.train_cfg is None:
            self.direction = ('a2b' if self.test_cfg is None else
                              self.test_cfg.get('direction', 'a2b'))
        else:
            self.direction = self.train_cfg.get('direction', 'a2b')
        self.step_counter = 0  # counting training steps

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if pretrained is not None:
            logger = logging.getLogger()
            logger.info(f'load model from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        # init GAN
        # self.G_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    # need correct
    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out_gen = self.G_head(x, img)
        return out_gen

    def forward(self, img):

        x = self.extract_feat(img)

        gen_out = self.G_head(x, img)  # the output of generators

        return gen_out
