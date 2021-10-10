from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, HEADS, LOSSES, SEGMENTORS, build_backbone, build_generator_head,
                      build_head, build_feature_selection, build_loss, build_segmentor, build_component,
                      build_generator,)
from .decode_heads import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .segmentors import *  # noqa: F401,F403
from .components import *
from .generator_head import *
from .feature_selection import *
from .generators import *
# from .discriminator import *
__all__ = [
    'BACKBONES', 'HEADS', 'LOSSES', 'SEGMENTORS', 'build_backbone',
    'build_head', 'build_loss', 'build_segmentor', 'build_component',
    'build_generator_head', 'build_feature_selection', 'build_generator',
]
