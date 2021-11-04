norm_cfg = dict(type='BN', requires_grad=True)
pretrained = '/opt/ml/trash-segmentation/mmsegmentation/pretrain/jx_vit_base_p16_224-80ecf9dd.pth'
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        norm_cfg=dict(type='LN', eps=1e-06),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        interpolate_mode='bicubic',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/opt/ml/trash-segmentation/mmsegmentation/pretrain/jx_vit_base_p16_224-80ecf9dd.pth'
        )),
    neck=dict(
        type='MultiLevelNeck',
        in_channels=[768, 768, 768, 768],
        out_channels=768,
        scales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'CustomDataset'
data_root = '/opt/ml/trash-segmentation/mmsegmentation/input/'
classes = [
    'Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
    'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
]
palette = [[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0],
           [64, 0, 128], [64, 0, 192], [192, 128, 64], [192, 192, 128],
           [64, 64, 128], [128, 0, 192]]
albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='Flip', p=1.0),
            dict(type='RandomRotate90', p=1.0),
            dict(type='Rotate', limit=(30, 30), p=1.0),
            dict(type='Rotate', limit=(-30, -30), p=1.0)
        ],
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=0.1,
        contrast_limit=0.15,
        p=0.5),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=15,
        sat_shift_limit=25,
        val_shift_limit=10,
        p=0.5),
    dict(
        type='CLAHE',
        clip_limit=4.0,
        tile_grid_size=(8, 8),
        always_apply=False,
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', p=1.0),
            dict(type='GaussianBlur', p=1.0),
            dict(type='MedianBlur', blur_limit=5, p=1.0),
            dict(type='MotionBlur', p=1.0)
        ],
        p=0.1)
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Flip', p=1.0),
                    dict(type='RandomRotate90', p=1.0),
                    dict(type='Rotate', limit=(30, 30), p=1.0),
                    dict(type='Rotate', limit=(-30, -30), p=1.0)
                ],
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.1,
                contrast_limit=0.15,
                p=0.5),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=10,
                p=0.5),
            dict(
                type='CLAHE',
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                always_apply=False,
                p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', p=1.0),
                    dict(type='GaussianBlur', p=1.0),
                    dict(type='MedianBlur', blur_limit=5, p=1.0),
                    dict(type='MotionBlur', p=1.0)
                ],
                p=0.1)
        ],
        keymap=dict(img='image', gt_semantic_seg='mask'),
        update_pad_shape=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
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
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5],
        flip=True,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        classes=[
            'Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64],
                 [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
                 [192, 192, 128], [64, 64, 128], [128, 0, 192]],
        type='CustomDataset',
        img_dir=
        '/opt/ml/segmentation/input/mmseg/images',
        ann_dir=
        '/opt/ml/segmentation/input/mmseg/annotations',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            dict(type='RandomFlip', prob=0),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Flip', p=1.0),
                            dict(type='RandomRotate90', p=1.0),
                            dict(type='Rotate', limit=(30, 30), p=1.0),
                            dict(type='Rotate', limit=(-30, -30), p=1.0)
                        ],
                        p=0.5),
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=0.1,
                        contrast_limit=0.15,
                        p=0.5),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=15,
                        sat_shift_limit=25,
                        val_shift_limit=10,
                        p=0.5),
                    dict(
                        type='CLAHE',
                        clip_limit=4.0,
                        tile_grid_size=(8, 8),
                        always_apply=False,
                        p=0.5),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', p=1.0),
                            dict(type='GaussianBlur', p=1.0),
                            dict(type='MedianBlur', blur_limit=5, p=1.0),
                            dict(type='MotionBlur', p=1.0)
                        ],
                        p=0.1)
                ],
                keymap=dict(img='image', gt_semantic_seg='mask'),
                update_pad_shape=True),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        classes=[
            'Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64],
                 [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
                 [192, 192, 128], [64, 64, 128], [128, 0, 192]],
        type='CustomDataset',
        img_dir=
        '/opt/ml/segmentation/input/mmseg/images/validation',
        ann_dir=
        '/opt/ml/segmentation/input/mmseg/annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        classes=[
            'Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64],
                 [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
                 [192, 192, 128], [64, 64, 128], [128, 0, 192]],
        type='CustomDataset',
        img_dir='/opt/ml/segmentation/input/mmseg/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5],
                flip=True,
                flip_direction=['horizontal', 'vertical'],
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='segmentation',
                entity='cv4',
                name='19_BEiT_allData_onlyVal'))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = '/opt/ml/trash-segmentation/mmsegmentation/output/19_beit_alldata/epoch_100.pth'
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_embed=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
checkpoint_config = dict(by_epoch=True, interval=5)
work_dir = './output/19_beit_alldata'
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr=1e-05)
runner = dict(type='EpochBasedRunner', max_epochs=120)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)
gpu_ids = range(0, 1)
