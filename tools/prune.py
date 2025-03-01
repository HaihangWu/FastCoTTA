import argparse
import os
import re

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import single_model_update, single_gpu_cotta,Efficient_adaptation,ETA_TENT,DPT,single_gpu_ours,single_gpu_AuxAdapt, single_gpu_RDumb
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from IPython import embed
from copy import deepcopy
import time
import numpy as np
from collections import deque


def load_rm_block_state_dict(model, raw_state_dict, rm_blocks):
    rm_block_info = [[] for _ in range(4)]
    for stage_index, block_index in (map(int, re.findall(r'\d+', rm_block)) for rm_block in rm_blocks):
        rm_block_info[stage_index].append(block_index)
    # has_count = [set(),set(),set(),set()]
    state_dict = model.state_dict()
    for raw_key in raw_state_dict.keys():
        key_items = raw_key.split('.')
        if 'backbone.block' in raw_key:
        #if key_items[0] == 'features':
            block = f'backbone.{key_items[1]}.{key_items[2]}'
            stage_index, block_index = map(int, re.findall(r'\d+', block))
            if block not in rm_blocks:
                key_items[2] = str(int(key_items[2]) - len([ rm_block_index for rm_block_index in rm_block_info[stage_index] if block_index>rm_block_index]))
                target_key = '.'.join(key_items)
                assert target_key in state_dict
                state_dict[target_key] = raw_state_dict[raw_key]

            # if block in rm_blocks:
            #     if block not in has_count[stage_index]:
            #         has_count[stage_index].add(block)
            #         rm_count[stage_index] += 1
            # else:

        else:
            assert raw_key in state_dict
            state_dict[raw_key] = raw_state_dict[raw_key]
    model.load_state_dict(state_dict)



def build_student(pruned_model_temp, pruned_block, state_dict_path='', cuda=True):

    pretrained_dict = torch.load(state_dict_path,map_location='cpu')
    pruned_model=load_rm_block_state_dict(pruned_model_temp,pretrained_dict['state_dict'],pruned_block)

    if cuda:
        pruned_model.cuda()
    return pruned_model

    # pruned_MACs, pruned_Params = compute_MACs_params(pruned_model, summary_data)
    # MACs_str = f'MACs={pruned_MACs:.3f}G'
    # Params_str = f'Params={pruned_Params:.3f}M'
    # print(f'=> pruned_model: {latency_str}, {MACs_str}, {Params_str}')

def eval_speed(model,test_time=500):
    print('=> testing latency. Please wait.')
    if not next(model.parameters()).is_cuda:
        model = model.cuda()
    data = torch.randn(1, 3, 1920//2, 1080//2)
    data = data.cuda()
    with torch.no_grad():
        output = model(data)

    test_begin=time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for i in range(test_time):
                output = model(data)
    total_time = time.time() - test_begin
    each_time = total_time / test_time

    latency = each_time * 1000
    latency_str = f'Lat={latency:.3f}ms'
    print(f'=> pruned_model: {latency_str}')
    return latency

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', default='work_dirs/res.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='mIoU',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--method',
        choices=['Source', 'BN', 'TENT', 'AuxAdapt', 'DPT', 'CoTTAETA', 'CoTTA', 'Ours','RDumb', 'VanillaETA'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--current_model_probs', default='empty', type=str,
                        help='EATA baseline')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if 'None' in args.eval:
        args.eval = None
    if args.eval and args.format_only:

        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # if 'AuxAdapt' in args.method:
    #     config_s= args.config.replace("base", "tiny") if 'segnext' in args.config else args.config.replace("b5", "b0")
    #     cfg_s = mmcv.Config.fromfile(config_s)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
        # if 'AuxAdapt' in args.method:
        #     cfg_s.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    #if args.aug_test:
    if True:
        for i in range(len(cfg.data.test.test_cases)):
            if cfg.data.test.test_cases[i].type in ['CityscapesDataset', 'ACDCDataset','KITTIDataset','NightCityDataset']:
                # hard code index
                cfg.data.test.test_cases[i].pipeline[1].img_ratios = [1.0]
                cfg.data.test.test_cases[i].pipeline[1].flip = False

            elif cfg.data.test.test_cases[i].type == 'ADE20KDataset':
                # hard code index
                cfg.data.test.test_cases[i].pipeline[1].img_ratios = [
                    0.75, 0.875, 1.0, 1.125, 1.25
                ]
                cfg.data.test.test_cases[i].pipeline[1].flip = True
            else:
                # hard code index
                cfg.data.test.test_cases[i].pipeline[1].img_ratios = [
                    0.5, 0.75, 1.0, 1.25, 1.5, 1.75
                ]
                cfg.data.test.test_cases[i].pipeline[1].flip = True

    # cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    #print(cfg)
    #datasets = [ build_dataset(cfg.data.test)]
    datasets = [ build_dataset(cfg.data.test.test_cases[i])for i in range(len(cfg.data.test.test_cases))]
    data_loaders = [build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False) for dataset in datasets]

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    #cfg.model.class_names=datasets[0].CLASSES
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    original_num_params = sum(p.numel() for p in model.parameters())


    pretrained_dict = torch.load(cfg.model.pretrained,map_location='cpu')
    #print(pretrained_dict.keys())
    #print("I'm printing the model",model.state_dict().keys())
    model.load_state_dict(pretrained_dict['state_dict'])
    # if hasattr(model, 'text_encoder'):
    #     model.text_encoder.init_weights()
    model.CLASSES = datasets[0].CLASSES
    model.PALETTE = datasets[0].PALETTE

    efficient_test = True #False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    model = MMDataParallel(model, device_ids=[0])

    anchor = deepcopy(model.state_dict()) #?
    anchor_model = deepcopy(model) #?

    i=0
    origin_lat=0
    for dataset, data_loader in zip(datasets, data_loaders):
        if i==0:
            img_id = 0
            model.eval()
            pred_begin = time.time()
            for j, data in enumerate(data_loader):
                with torch.no_grad():
                    result_ori, probs, preds = model(return_loss=False, **data)
            origin_lat = time.time() - pred_begin
        i=i+1


    original_depth=[3, 6, 40, 3]
    prunable_blocks = [
        'backbone.block' + str(stage_index) + '.' + str(block_index)
        for stage_index, block_num in enumerate(original_depth)
        for block_index in range(block_num)
    ]
    Model_capacity_gap = []
    latency_time_saving = []




    for block_index, pruned_block in enumerate(prunable_blocks):
        # build the model and load checkpoint
        stage_index, block_index = map(int, re.findall(r'\d+', pruned_block))
        cfg.model.backbone.depths = [original_depth[i] - 1 if stage_index == i else original_depth[i] for i in
                                     range(4)]
        pruned_model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        #pruned_model = build_student(pruned_model_temp, [pruned_block], state_dict_path=cfg.model.pretrained, cuda=True)
        pruned_num_params = sum(p.numel() for p in pruned_model.parameters())
        Model_capacity_gap.append((original_num_params-pruned_num_params)/original_num_params)

        i = 0
        pruned_lat = 0
        for dataset, data_loader in zip(datasets, data_loaders):
            if i == 0:
                img_id = 0
                pruned_model.eval()
                pred_begin = time.time()
                for j, data in enumerate(data_loader):
                    with torch.no_grad():
                        result_ori, probs, preds = model(return_loss=False, **data)
                pruned_lat = time.time() - pred_begin
            i = i + 1

        lat_reduction = (origin_lat - pruned_lat) / origin_lat
        latency_time_saving.append(lat_reduction)
    print("I'm printing prunable_blocks",prunable_blocks)
    print("I'm printing model capacity gap",Model_capacity_gap)
    print("I'm printing latency time saving",latency_time_saving)

    # blocks_importance=[]
    # criterion = torch.nn.MSELoss(reduction='mean').cuda()
    # t_features=None
    #
    #
    #
    # total_predict_time=0
    # total_processed_frame=0
    #
    # for dataset, data_loader in zip(datasets, data_loaders):
    #     # j=j+1
    #     pred_begin = time.time()
    #
    #
    #     #######################################Create pruned model######################################
    #     prune_start = time.time()
    #     prune_loader=data_loader[:10]
    #     feature_maps_origin = []
    #     for i, data in enumerate(prune_loader):
    #         with torch.no_grad():
    #             result_ori, probs, preds = model(return_loss=False, **data)
    #             result = [preds[0][0].astype(np.int64)]
    #             if isinstance(result, list):
    #                 result = [np2tmp(_) for _ in result]
    #                 feature_maps_origin.extend(result)
    #             else:
    #                 result = np2tmp(result)
    #                 feature_maps_origin.append(result)
    #
    #     for block_index, pruned_block in enumerate(prunable_blocks):
    #         # build the model and load checkpoint
    #         stage_index, block_index = map(int, re.findall(r'\d+', pruned_block))
    #         cfg.model.backbone.depths = [original_depth[i] - 1 if stage_index == i else original_depth[i] for i in
    #                                      range(4)]
    #         pruned_model_temp = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    #         pruned_model = build_student(pruned_model_temp, [pruned_block],  state_dict_path=cfg.model.pretrained, cuda=True)
    #         pruned_num_params = sum(p.numel() for p in pruned_model.parameters())
    #
    #         loss = 0
    #         feature_maps_prune = []
    #         for i, data in enumerate(prune_loader):
    #             with torch.no_grad():
    #                 result_ori, probs, preds = pruned_model(return_loss=False, **data)
    #                 result = [preds[0][0].astype(np.int64)]
    #                 if isinstance(result, list):
    #                     result = [np2tmp(_) for _ in result]
    #                     feature_maps_prune.extend(result)
    #                 else:
    #                     result = np2tmp(result)
    #                     feature_maps_prune.append(result)
    #             loss = loss+criterion(feature_maps_prune, feature_maps_origin).data.item()
    #         blocks_importance.append(loss * Model_capacity_gap[block_index] / latency_time_saving[block_index])
    #
    #     paired_lists = zip(blocks_importance, prunable_blocks)
    #     sorted_lists = sorted(paired_lists, key=lambda x: x[0])
    #     sorted_blocks_importance, sorted_prunable_blocks = zip(*sorted_lists)
    #     pruned_block = sorted_prunable_blocks[:args.num_rm_blocks]
    #     print(f'pruning time: {(time.time() - prune_start):.6f}/block importance: {blocks_importance}')
    #
    #     pruned_block_info = [[] for _ in range(4)]
    #     for stage_index, block_index in (map(int, re.findall(r'\d+', rm_block)) for rm_block in pruned_block):
    #         pruned_block_info[stage_index].append(block_index)
    #     cfg.model.backbone.depths = [original_depth[i] - len(pruned_block_info[i]) for i in range(4)]
    #
    #     pruned_model_temp = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    #     pruned_model = build_student(pruned_model_temp, pruned_block, state_dict_path=cfg.model.pretrained,cuda=True)
    #
    #
    #     total_predict_time = total_predict_time+time.time()-pred_begin
    #     total_processed_frame=total_processed_frame+len(data_loader)
    #
    #     rank, _ = get_dist_info()
    #     if rank == 0:
    #         if args.out:
    #             print(f'\nwriting results to {args.out}')
    #             mmcv.dump(outputs, args.out)
    #         kwargs = {} if args.eval_options is None else args.eval_options
    #         if args.format_only:
    #             dataset.format_results(outputs, **kwargs)
    #         if args.eval:
    #                 dataset.evaluate(outputs, args.eval, **kwargs)
    # print("total avg pred time:%.3f seconds; " % (total_predict_time / total_processed_frame))

if __name__ == '__main__':
    main()


# def create_ema_model(model):
#     ema_model = deepcopy(model)#get_model(args.model)(num_classes=num_classes)
#
#     for param in ema_model.parameters():
#         param.detach_()
#     mp = list(model.parameters())
#     mcp = list(ema_model.parameters())
#     n = len(mp)
#     for i in range(0, n):
#         mcp[i].data[:] = mp[i].data[:].clone()
#     #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
#     #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
#     return ema_model
#
# def update_ema_variables(ema_model, model, alpha_teacher, iteration=None):
#     # Use the "true" average until the exponential average is more correct
#     if iteration:
#         alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
#
#     if True:
#         for ema_param, param in zip(ema_model.parameters(), model.parameters()):
#             #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
#             ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
#     return ema_model

    # current_model_probs=None
    # loss = 0
    # for i in range(10):
    #     print("revisit times:",i)
    #     j=0

    # ema_model = create_ema_model(model) #?
    # for name, param in anchor_model.named_parameters():
    #     if "DSP" in name or "DAP" in name:
    #             param.data = torch.zeros_like(param)
    # print([param.data for name, param in anchor_model.named_parameters() if "DSP" in name or "DAP" in name])
    # frame_passed=0
    # total_predict_time=0
    #
    #
    # ldelta=0

    # file_name=args.method
    # model_name='segformer' if 'segformer' in args.config else ('segnext' if 'segnext' in args.config else '')
    # dataset_name='acdc' if 'acdc' in args.config else ('night' if 'night' in args.config  else '')
    # print(file_name+ '_' + model_name+ '_'+ dataset_name)

    # if 'TENT' in args.method or 'VanillaETA' in args.method:
    #     for name, param in model.named_parameters():
    #         if ("norm" in name or "bn" in name or "ln" in name or "BatchNorm" in name):
    #                 param.requires_grad = True
    #         else:
    #             param.requires_grad = False

    #checkpoint = load_checkpoint(model, cfg.model.pretrained, map_location='cpu')
    # model.CLASSES = checkpoint['meta']['CLASSES']
    # model.PALETTE = checkpoint['meta']['PALETTE']


# if 'Source' in args.method or 'BN' in args.method or 'TENT' in args.method or 'VanillaETA' in args.method:
#     cfg.data.test.test_cases[i].pipeline[1].img_ratios = [1.0]
#     cfg.data.test.test_cases[i].pipeline[1].flip = False
# elif 'AuxAdapt' in args.method or 'Ours' in args.method:
#     cfg.data.test.test_cases[i].pipeline[1].img_ratios = [1.0, 2.0]9
#     cfg.data.test.test_cases[i].pipeline[1].flip = False


#################################################################################################


# if 'Source' in args.method or 'BN' in args.method or 'TENT' in args.method:
#     outputs, frame_passed = single_model_update(model, data_loader, args, efficient_test,frame_passed)
#
# elif 'VanillaETA' in args.method:
#     outputs,frame_passed = ETA_TENT(model,data_loader,current_model_probs,efficient_test,anchor_model,frame_passed)
#
# elif 'Ours' in args.method:
#     outputs,frame_passed,domains_detections = single_gpu_ours(model, data_loader, args.show, args.show_dir,
#                               efficient_test,anchor, ema_model, anchor_model,frame_passed, domains_detections,i*4+j)