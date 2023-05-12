# dataset settings
dataset_type_night = 'NightCityDataset'
data_root_night = '/data/gpfs/projects/punim0512/data/NightCity/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline_night = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
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
    test = dict(
    test_cases=[
        dict(
            type=dataset_type_night,
            data_root=data_root_night,
            img_dir='rgb_annon/train/Chicago',
            ann_dir='gt/train/Chicago',
            pipeline=test_pipeline_night),
        dict(
            type=dataset_type_night,
            data_root=data_root_night,
            img_dir='rgb_annon/train/Dubai',
            ann_dir='gt/train/Dubai',
            pipeline=test_pipeline_night),
        dict(
            type=dataset_type_night,
            data_root=data_root_night,
            img_dir='rgb_annon/train/Helsinki',
            ann_dir='gt/train/Helsinki',
            pipeline=test_pipeline_night),
        dict(
            type=dataset_type_night,
            data_root=data_root_night,
            img_dir='rgb_annon/train/HK',
            ann_dir='gt/train/HK',
            pipeline=test_pipeline_night),
        dict(
            type=dataset_type_night,
            data_root=data_root_night,
            img_dir='rgb_annon/train/LA',
            ann_dir='gt/train/LA',
            pipeline=test_pipeline_night),
        dict(
            type=dataset_type_night,
            data_root=data_root_night,
            img_dir='rgb_annon/train/London',
            ann_dir='gt/train/London',
            pipeline=test_pipeline_night),
        dict(
            type=dataset_type_night,
            data_root=data_root_night,
            img_dir='rgb_annon/train/Melbourne',
            ann_dir='gt/train/Melbourne',
            pipeline=test_pipeline_night),
        dict(
            type=dataset_type_night,
            data_root=data_root_night,
            img_dir='rgb_annon/train/Toronto',
            ann_dir='gt/train/Toronto',
            pipeline=test_pipeline_night),
        dict(
            type=dataset_type_night,
            data_root=data_root_night,
            img_dir='rgb_annon/train/Youtube',
            ann_dir='gt/train/Youtube',
            pipeline=test_pipeline_night),
        dict(
            type=dataset_type_night,
            data_root=data_root_night,
            img_dir='rgb_annon/train/Nagoya',
            ann_dir='gt/train/Nagoya',
            pipeline=test_pipeline_night),
        dict(
            type=dataset_type_night,
            data_root=data_root_night,
            img_dir='rgb_annon/train/Tokyo',
            ann_dir='gt/train/Tokyo',
            pipeline=test_pipeline_night),
        dict(
            type=dataset_type_night,
            data_root=data_root_night,
            img_dir='rgb_annon/train/NYC',
            ann_dir='gt/train/NYC',
            pipeline=test_pipeline_night),
        dict(
            type=dataset_type_night,
            data_root=data_root_night,
            img_dir='rgb_annon/train/Seoul',
            ann_dir='gt/train/Seoul',
            pipeline=test_pipeline_night)
    ])

)
