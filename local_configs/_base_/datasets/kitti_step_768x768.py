# dataset settings
dataset_type = 'KITTIDataset'
data_root = '/data/gpfs/projects/punim0512/data/kitti_step/'
#data_root =  '/data/gpfs/projects/punim0512/data/test/kitti_step/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1242,375), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1242//2, 375//2),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='rgbd/train/',
            ann_dir='gt/train/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgbd/train/',
        ann_dir='gt/train/',
        pipeline=test_pipeline),
    test=dict(
        test_cases=[dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgbd/train/00',
        ann_dir='gt/train/00',
        pipeline=test_pipeline),
        dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgbd/train/01',
        ann_dir='gt/train/01',
        pipeline=test_pipeline),
        dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgbd/train/03',
        ann_dir='gt/train/03',
        pipeline=test_pipeline),
        dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgbd/train/03',
        ann_dir='gt/train/03',
        pipeline=test_pipeline),
        dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgbd/train/05',
        ann_dir='gt/train/05',
        pipeline=test_pipeline),
        dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgbd/train/09',
        ann_dir='gt/train/09',
        pipeline=test_pipeline),
        dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgbd/train/11',
        ann_dir='gt/train/11',
        pipeline=test_pipeline),
        dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgbd/train/12',
        ann_dir='gt/train/12',
        pipeline=test_pipeline),
        dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='rgbd/train/15',
            ann_dir='gt/train/15',
            pipeline=test_pipeline),
        dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='rgbd/train/17',
            ann_dir='gt/train/17',
            pipeline=test_pipeline),
        dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='rgbd/train/19',
            ann_dir='gt/train/19',
            pipeline=test_pipeline),
        dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='rgbd/train/20',
            ann_dir='gt/train/20',
            pipeline=test_pipeline)
        ])
)
