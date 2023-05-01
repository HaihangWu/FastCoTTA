_base_ = [
    '../_base_/datasets/cityscapes_768x768.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
crop_size = (768, 768)
model = dict(
    type='SETRLanguage',
    pretrained='../mmsegmentation/setr_pup_vit-large_8x1_768x768_80k_cityscapes.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(768, 768),
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        out_indices=(9, 14, 19, 23),
        drop_rate=0.1,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=True,
        interpolate_mode='bilinear',
    ),
    decode_head=dict(
        type='SETRUPHead',
        in_channels=1024,
        channels=256,
        in_index=3,
        num_classes=19,
        dropout_ratio=0,
        norm_cfg=norm_cfg,
        num_convs=4,
        up_scale=2,
        kernel_size=3,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    text_encoder=dict(
        type='CLIPTextEncoder',
        pretrained='pretrained/ViT-B-16.pt',
        context_length=77,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    text_decoder=dict(
        type='TextSETRHead',
        in_channels=1024,
        channels=512,
        in_index=3,
        num_classes=19,
        dropout_ratio=0,
        norm_cfg=norm_cfg,
        num_convs=4,
        up_scale=2,
        kernel_size=3,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    ft_model=False,
    include_key='linear_pred',
    # load_text_embedding='configs/_base_/datasets/text_embedding/voc12_single.npy'
)

optimizer = dict(
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

data = dict(samples_per_gpu=1)
