checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # Wandb Logger Hook
        dict(type='WandbLoggerHook',
            init_kwargs=dict(
                project='segmentation',
                entity='cv4',
                name='zzin'
            ))
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
