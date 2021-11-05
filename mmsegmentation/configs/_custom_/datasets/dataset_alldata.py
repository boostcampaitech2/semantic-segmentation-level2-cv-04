# dataset settings
dataset_type = 'CustomDataset'
data_root = '/opt/ml/segmentation/input/mmseg/'

# class settings
classes = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic','Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
palette = [
    [0, 0, 0],
    [192, 0, 128], [0, 128, 192], [0, 128, 64],
    [128, 0, 0], [64, 0, 128], [64, 0, 192],
    [192, 128, 64], [192, 192, 128], [64, 64, 128], [128, 0, 192]
    ]

# Albumentations transforms settings
albu_train_transforms = [
    dict(
    type='OneOf',
    transforms=[
        dict(type='Flip', p=1.0),
        dict(type='RandomRotate90', p=1.0),
        dict(type='Rotate', limit=(30,30), p=1.0),
        dict(type='Rotate', limit=(-30,-30), p=1.0),
    ], p=0.5),

    dict(
    type='OneOf',
    transforms=[
        dict(type='RandomBrightnessContrast',brightness_limit=0.1, contrast_limit=0.15, p=1.0),
        dict(type='CLAHE', clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=1.0),
        dict(type='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=1.0),
    ], p=0.5),
    
    dict(
    type='OneOf',
    transforms=[
        dict(type='Blur', p=1.0),
        dict(type='GaussianBlur', p=1.0),
        dict(type='MedianBlur', blur_limit=5, p=1.0),
        dict(type='MotionBlur', p=1.0)
    ], p=0.1)
]

# set normalize value
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        keymap=dict(img='image', gt_semantic_seg='mask'),
        update_pad_shape=True,
        ),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'), 
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5],  # for TTA
        flip=True,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        img_dir=data_root + "images",
        ann_dir=data_root + "annotations",
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        img_dir=data_root + "images/validation",
        ann_dir=data_root + "annotations/validation",
        pipeline=valid_pipeline),
    test=dict(
        classes=classes,
        palette=palette,
        type=dataset_type,
        img_dir=data_root + "test",
        pipeline=test_pipeline))