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
import itertools
from collections import deque
import tempfile

def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name

def load_rm_block_state_dict(model, raw_state_dict, rm_blocks):
    rm_block_info = [[] for _ in range(4)]
    for stage_index, block_index in (map(int, re.findall(r'\d+', rm_block)) for rm_block in rm_blocks):
        rm_block_info[stage_index-1].append(block_index)
    # has_count = [set(),set(),set(),set()]
    state_dict = model.state_dict()
    for raw_key in raw_state_dict.keys():
        key_items = raw_key.split('.')
        if 'backbone.block' in raw_key:
        #if key_items[0] == 'features':
            block = f'backbone.{key_items[1]}.{key_items[2]}'
            stage_index, block_index = map(int, re.findall(r'\d+', block))
            if block not in rm_blocks:
                try:
                    key_items[2] = str(int(key_items[2]) - len([ rm_block_index for rm_block_index in rm_block_info[stage_index-1] if block_index>rm_block_index]))
                except:
                    print(f"rm_block_info: {rm_block_info}")
                    raise
                target_key = '.'.join(key_items)
                try:
                    assert target_key in state_dict
                except AssertionError:
                    print(f"target Key '{target_key}' not found in state_dict.")
                    print(f"raw keys: {raw_key}")
                    print(f"block: {block}")
                    print(f"removed blocks: {rm_blocks}")
                    print(f"Available keys: {list(state_dict.keys())}")
                    raise
                state_dict[target_key] = raw_state_dict[raw_key]
        else:
            assert raw_key in state_dict
            state_dict[raw_key] = raw_state_dict[raw_key]
    model.load_state_dict(state_dict)



def build_student(pruned_model_temp, pruned_block, state_dict_path='', cuda=True):

    pretrained_dict = torch.load(state_dict_path,map_location='cpu')
    load_rm_block_state_dict(pruned_model_temp,pretrained_dict['state_dict'],pruned_block)

    # if cuda:
    #     pruned_model_temp.cuda()
    return pruned_model_temp


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('config_test', help='test config file path')
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
    parser.add_argument('--num_rm_blocks', type=int, default=0)
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
    cfg_test=mmcv.Config.fromfile(args.config_test)
    # if 'AuxAdapt' in args.method:
    #     config_s= args.config.replace("base", "tiny") if 'segnext' in args.config else args.config.replace("b5", "b0")
    #     cfg_s = mmcv.Config.fromfile(config_s)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
        cfg_test.merge_from_dict(args.options)
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

        for i in range(len(cfg_test.data.test.test_cases)):
            if cfg_test.data.test.test_cases[i].type in ['CityscapesDataset', 'ACDCDataset','KITTIDataset','NightCityDataset']:
                # hard code index
                cfg_test.data.test.test_cases[i].pipeline[1].img_ratios = [1.0]
                cfg_test.data.test.test_cases[i].pipeline[1].flip = False

    # cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg_test.data.test.test_mode = True

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

    datasets_test = [ build_dataset(cfg_test.data.test.test_cases[i])for i in range(len(cfg_test.data.test.test_cases))]
    data_loaders_test = [build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg_test.data.workers_per_gpu,
        dist=distributed,
        shuffle=False) for dataset in datasets_test]

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    cfg_test.model.train_cfg = None
    #cfg.model.class_names=datasets[0].CLASSES
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    original_model_size = sum(p.numel() for p in model.parameters())

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

    original_depth=[3, 6, 40, 3]
    prunable_blocks = [
        'backbone.block' + str(stage_index+1) + '.' + str(block_index)
        for stage_index, block_num in enumerate(original_depth)
        for block_index in range(block_num)
    ]
    Model_capacity_gap = [0.003722, 0.003722, 0.003722, 0.005507, 0.005507, 0.005507, 0.005507, 0.005507, 0.005507, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.019576, 0.0375, 0.0375, 0.0375]
    latency_time_saving = [0.017562353105231077, 0.019136228420689162, 0.020002722264649612, 0.018709442052548004, 0.018877205451712575, 0.020109581577526132, 0.020273178607898828, 0.019398023008389952, 0.019284486776599764, 0.020015471710780965, 0.018897472243080422, 0.01992021743372387, 0.019118711790352173, 0.018028532222078217, 0.01988822401294096, 0.020082934698669272, 0.019516533848754812, 0.019787451532411026, 0.019399278283450852, 0.018572041289471843, 0.02021899435591114, 0.019048126708081634, 0.019450730255818803, 0.01936838278131095, 0.020219180322586832, 0.018404753535843168, 0.018571647898427114, 0.019873139254516826, 0.019004263606594667, 0.019778078096699868, 0.018692958967773978, 0.018512077765408905, 0.020084368787841412, 0.019200519246244087, 0.018803734309686633, 0.019574079806033793, 0.020108558760809844, 0.019279837609707546, 0.019433406744721958, 0.01994351691241833, 0.018699989938627905, 0.019010325404965674, 0.019501985532664395, 0.018220768120506978, 0.01962777053107446, 0.01944367067470708, 0.017155121848295034, 0.019476890760293093, 0.01907537082607003, 0.01775815813332017, 0.01942708387774854, 0.018452418225335075]
    # latency_time_saving = [0.046348852953935774, 0.050502471499634424, 0.05278923771590252, 0.04937613845638028, 0.04981888329087601, 0.053071250413722446, 0.053502999773219505, 0.05119337429476534, 0.05089374052241965, 0.052822884813216044, 0.04987236943406696, 0.052571498996401, 0.05045624331376019, 0.04757914750543912, 0.052487065074222385, 0.05300092656461079, 0.05150613656987767, 0.05222111615189408, 0.05119668709452888, 0.04901352373582334, 0.05336000196918103, 0.05026996203474842, 0.05133247413213046, 0.051115150692371374, 0.05336049275433119, 0.04857203417873089, 0.04901248553646724, 0.05244725484800403, 0.050154202806543556, 0.052196378692691066, 0.049332637903359795, 0.0488552738394262, 0.053004711273172564, 0.05067214158894864, 0.04962498540386087, 0.05165800625932279, 0.05306855109539659, 0.05088147089366577, 0.05128675560775832, 0.05263298871280986, 0.04935119335730605, 0.05017020051480342, 0.051467742070053994, 0.048086477778060084, 0.051799701595078164, 0.05131384317277649, 0.04527412785685564, 0.05140151438931094, 0.05034186205924615, 0.04686560252616932, 0.05127006891265304, 0.04869782618798604]
    # latency_time_saving = [
    #     0.049880187, 0.049880187, 0.049880187, 0.051309398, 0.051309398, 0.051309398,
    #     0.051309398, 0.051309398, 0.051309398, 0.050869306, 0.050869306, 0.050869306,
    #     0.050869306, 0.050869306, 0.050869306, 0.050869306, 0.050869306, 0.050869306,
    #     0.050869306, 0.050869306, 0.050869306, 0.050869306, 0.050869306, 0.050869306,
    #     0.050869306, 0.050869306, 0.050869306, 0.050869306, 0.050869306, 0.050869306,
    #     0.050869306, 0.050869306, 0.050869306, 0.050869306, 0.050869306, 0.050869306,
    #     0.050869306, 0.050869306, 0.050869306, 0.050869306, 0.050869306, 0.050869306,
    #     0.050869306, 0.050869306, 0.050869306, 0.050869306, 0.050869306, 0.050869306,
    #     0.050869306, 0.050869306, 0.048944499, 0.048944499, 0.048944499]

    criterion = torch.nn.MSELoss(reduction='mean').cuda()


    model.eval()
    #######################################test the original model######################################
    # dataset_time_full=[]
    # for dataset, data_loader in zip(datasets_test, data_loaders_test):
    #     pred_begin_full = time.time()
    #     for i, data in enumerate(data_loader):
    #         with torch.no_grad():
    #             result, probs, preds = model(return_loss=False, **data)
    #     pred_time_full = time.time() - pred_begin_full
    #     dataset_time_full.append(pred_time_full)
    #####################################################################################################
    for dataset, data_loader in zip(datasets, data_loaders):
        prune_loader = []
        finetune_loader = []
        for i, data in enumerate(data_loader):
            if i<=9:
                prune_loader.append(data)
            if i<=99:
                finetune_loader.append(data)
            else:
                break
        #######################################Create pruned model######################################
        prune_start = time.time()
        feature_maps_origin = []
        for i, data in enumerate(prune_loader):
            with torch.no_grad():
                result_ori, probs, preds = model(return_loss=False, **data)
                feature_maps_origin.append(probs)
                # print(type(probs))
                # print(probs.shape)

        blocks_importance=[]
        for _, pruned_block in enumerate(prunable_blocks):
            # build the model and load checkpoint
            stage_index, block_index = map(int, re.findall(r'\d+', pruned_block))
            cfg.model.backbone.depths = [original_depth[i] - 1 if stage_index == (i+1) else original_depth[i] for i in
                                         range(4)]
            pruned_model_temp = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
            pruned_model = build_student(pruned_model_temp, [pruned_block],  state_dict_path=cfg.model.pretrained, cuda=True)
            pruned_model.CLASSES = datasets[0].CLASSES
            pruned_model.PALETTE = datasets[0].PALETTE
            pruned_model = MMDataParallel(pruned_model, device_ids=[0])

            pruned_model.eval()
            loss_prune = 0
            for i, data in enumerate(prune_loader):
                with torch.no_grad():
                    result_ori, probs, preds = pruned_model(return_loss=False, **data)
                loss_prune = loss_prune+criterion(probs, feature_maps_origin[i]).data.item()
                #print(f"loss at the {i}th time:{loss}")
            blocks_importance.append(loss_prune * Model_capacity_gap[block_index] / latency_time_saving[block_index])

        paired_lists = zip(blocks_importance, prunable_blocks)
        sorted_lists = sorted(paired_lists, key=lambda x: x[0])
        sorted_blocks_importance, sorted_prunable_blocks = zip(*sorted_lists)
        pruned_block = sorted_prunable_blocks[:args.num_rm_blocks]
        # print(f"sorted_prunable_blocks:{sorted_prunable_blocks}")
        indices = [prunable_blocks.index(block) for block in pruned_block]
        total_latency_saving = sum(latency_time_saving[i] for i in indices)*100
        print(f'pruning time: {(time.time() - prune_start):.6f}') #/block importance: {blocks_importance}

        pruned_block_info = [[] for _ in range(4)]
        for stage_index, block_index in (map(int, re.findall(r'\d+', rm_block)) for rm_block in pruned_block):
            pruned_block_info[stage_index-1].append(block_index)
        cfg.model.backbone.depths = [original_depth[i] - len(pruned_block_info[i]) for i in range(4)]

        pruned_model_temp = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        pruned_model = build_student(pruned_model_temp, pruned_block, state_dict_path=cfg.model.pretrained,cuda=True)
        pruned_model.CLASSES = datasets[0].CLASSES
        pruned_model.PALETTE = datasets[0].PALETTE
        pruned_model = MMDataParallel(pruned_model, device_ids=[0])
        pruned_model_size = sum(p.numel() for p in pruned_model.parameters())

        print(f"Original Model Size: {original_model_size} parameters； pruned model size: {pruned_model_size}; latency time saving: {total_latency_saving}%; backbone is  {cfg.model.backbone.depths}")
        #######################################finetune the pruned model######################################
        # pruned_model.eval() #？
        # param_list = []
        # for name, param in pruned_model.named_parameters():
        #     if param.requires_grad:
        #         param_list.append(param)
        #         # print(name)
        #     else:
        #         param.requires_grad = False
        # optimizer = torch.optim.Adam(param_list, lr=0.00006 / 8, betas=(0.9, 0.999))  # for segformer
        # t_features=[]
        #
        # finetune_start=time.time()
        # for finetune_iter in range(10):
        #     total_loss = 0
        #     for i, data in enumerate(finetune_loader):
        #         if finetune_iter == 0:
        #             _, t_feature, _ = model(return_loss=False, **data)
        #             t_features.append(t_feature.detach())
        #         else:
        #             t_feature = t_features[i]
        #         optimizer.zero_grad()
        #         _, s_feature, _ = pruned_model(return_loss=False, **data)
        #         loss_finetne=torch.mean(torch.square(t_feature - s_feature))
        #         loss_finetne.backward()
        #         optimizer.step()
        #         total_loss = total_loss+ loss_finetne.item()
        #     print(f"total_loss{total_loss}")
        #
        # print(f'finetuning time: {(time.time() - finetune_start):.6f}')

        #######################################test the pruned model######################################
    pruned_model.eval()  # ？
    model.eval()
    dataset_index=0
    for dataset, data_loader in zip(datasets_test, data_loaders_test):
        outputs = []
        param_list = []
        for name, param in pruned_model.named_parameters():
            if param.requires_grad:
                param_list.append(param)
            else:
                param.requires_grad = False
                print(f"Parameter {name} does not require grad")
        optimizer = torch.optim.Adam(param_list, lr=0.00006 / 8, betas=(0.9, 0.999))  # for segformer,segnext

        pred_time = 0
        pred_time_full = 0
        data_load_begin = time.time()
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                data_load_time =time.time() - data_load_begin

                pred_begin_full = time.time()
                result_full, probs_full, preds_full = model(return_loss=False, **data)
                pred_time_full += time.time() - pred_begin_full + data_load_time

                pred_begin = time.time()
                result, probs, preds = pruned_model(return_loss=False, **data)

                img_id = 0
                if isinstance(result, list):
                    loss = pruned_model.forward(return_loss=True, img=data['img'][img_id],
                                             img_metas=data['img_metas'][img_id].data[0],
                                             gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(
                                                 0))
                    if efficient_test:
                        result = [np2tmp(_) for _ in result]
                    outputs.extend(result)
                else:
                    loss = pruned_model(return_loss=True, img=data['img'][img_id],
                                     img_metas=data['img_metas'][img_id].data[0], gt_semantic_seg=result)
                    if efficient_test:
                        result = np2tmp(result)
                    outputs.append(result)

                loss_value = loss["decode.loss_seg"]
                print(loss, loss_value.requires_grad, loss_value.grad_fn)
                torch.mean(loss["decode.loss_seg"]).backward()
                optimizer.step()
                optimizer.zero_grad()

                pred_time += time.time() - pred_begin + data_load_time
                data_load_begin = time.time()

        print(f"pred time for pruned model: {pred_time}; pred time for full model: {pred_time_full}; latency saving: {(pred_time_full-pred_time)/pred_time_full*100}%")
        dataset_index = dataset_index+1

        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(outputs, **kwargs)
            if args.eval:
                    dataset.evaluate(outputs, args.eval, **kwargs)

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