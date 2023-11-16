_base_ = [
    '../_base_/models/mscan.py',
    '../_base_/datasets/cityscapes_1024x1024_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_adamw.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
prompt_config=dict(NUM_TOKENS = 6000,LOCATION = "random")
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='/data/gpfs/projects/punim0512/Haihangw-Projects/segformer/segnext_base_1024x1024_city_160k.pth',
    backbone=dict(
        prompt_config=prompt_config,
        embed_dims=[64, 128, 320, 512],
        depths=[3, 3, 12, 3],
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/mscan_b.pth'),
        drop_path_rate=0.1),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=512,
        ham_channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# data
data = dict(samples_per_gpu=6)
evaluation = dict(interval=170000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=2000)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006/8*6, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

#checkpoint_config = dict(by_epoch=True)
