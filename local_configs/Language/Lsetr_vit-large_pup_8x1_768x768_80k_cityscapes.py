_base_ = [
    '../_base_/models/setr_pup.py', '../_base_/datasets/cityscapes_768x768.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (768, 768)
model = dict(
    pretrained='../mmsegmentation/setr_pup_vit-large_8x1_768x768_80k_cityscapes.pth',
    backbone=dict(
        drop_rate=0.,
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrain/vit_large_p16.pth')),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(512, 512)),
    ft_model=False,
    include_key='linear_pred',
    # load_text_embedding='configs/_base_/datasets/text_embedding/voc12_single.npy'
)

optimizer = dict(
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))

data = dict(samples_per_gpu=1)
