# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder_joint',
    pretrained=None,
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        in_channels=1,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    backbone_gan=dict(
        type='UNet_gan',
        in_channels=1,
        out_indices=(0, 1, 2),
        u_net_channels=(64, 128, 256),
        norm_cfg=None,
        norm_eval=False,
        style='pytorch',),
    feature_selection=dict(
        type='Basic_fs',
        in_channels=[256, 512, 1024, 2048],
        inu_channels=[64, 128, 256],
        adaptive_channel=128,
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    generator_head=dict(
        type='BaseGHead',
        in_channels=[64, 128, 256],
        out_channels=1,
        norm_cfg=None,
    ),
)

discriminator = dict(
    type='Discriminator',
    in_channels=1,
    fc_in_channels=14 * 14,
    fc_out_channels=1,
    norm_cfg=dict(type='IN')
)
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
