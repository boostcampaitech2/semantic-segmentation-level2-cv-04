# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    #betas=(0.9, 0.999),
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
