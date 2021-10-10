# dataset settings
dataset_type = 'ScoliosisDataset3'
dataset_gan_type = 'ScoliosisDataset_gan_m'
data_root = '/mnt/sd2/Semantic_Seg/mmsegmentation_for_3classes/data/scoliosis3classes'
img_norm_cfg = dict(
    mean=[75.99], std=[84.15], to_rgb=False)
crop_size = (448, 448)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 2048), ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.8),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
train_gan_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='Resize', img_scale=(128, 128), ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', crop_size=(128, 128), cat_max_ratio=1),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(128, 128)),
    # dict(type='DefaultFormatBundle'),
    dict(type='ToTensor', keys=['img'])
    # dict(type='Collect', keys=['img', 'gt_semantic_seg'],
    #      ),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(395, 220),
        # img_ratios=[0.8, 1.0, 1.2],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset/data',
        ann_dir='groundtruth/gt',
        split='train3.txt',
        pipeline=train_pipeline),
    train_gan=dict(
        type=dataset_gan_type,
        data_root=data_root,
        img_dir='gan_dataset',
        cln_dir='train_cln',
        cor_dir='train_cor',
        # split='train.txt',
        pipeline=train_gan_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset/data',
        ann_dir='groundtruth/gt',
        split='test3.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset/data',
        ann_dir='groundtruth/gt',
        split='test3.txt',
        pipeline=test_pipeline))

