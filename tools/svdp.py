import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import single_model_update, single_gpu_cotta,Efficient_adaptation,ETA_TENT,DPT,single_gpu_ours,single_gpu_AuxAdapt, single_gpu_RDumb,single_gpu_svdp
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from IPython import embed
from copy import deepcopy
import time
from collections import deque
# import wandb
import numpy as np
import random

def set_random_seed(seed=1, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_ema_model(model):
    ema_model = deepcopy(model) # get_model(args.model)(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    if True:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

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
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--model_lr', type=float, default=3e-4)
    parser.add_argument('--prompt_lr', type=float, default=1e-4)
    parser.add_argument('--prompt_sparse_rate', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ema_rate', type=float, default=0.999)
    parser.add_argument('--ema_rate_prompt', type=float, default=0.999)
    # parser.add_argument('--wandb_login', type=str)
    # parser.add_argument('--wandb_project', type=str)
    # parser.add_argument('--wandb_name', type=str, default="debug")


    parser.add_argument('--current_model_probs', default='empty', type=str,
                        help='EATA baseline')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    set_random_seed(seed=args.seed)
    # wandb.init(
    #     name=args.wandb_name,
    #     project=args.wandb_project,
    #     entity=args.wandb_login,
    #     mode="online",
    #     save_code=True,
    #     config=args,
    # )
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>init param>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print("model_lr", args.model_lr)
    print("prompt_lr", args.prompt_lr)
    print("prompt_sparse_rate", args.prompt_sparse_rate)
    print("ema_rate", args.ema_rate)
    print("seed", args.seed)
    print("scale", args.scale)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>over param>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

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
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    #if args.aug_test:
    if True:
        for i in range(len(cfg.data.test.test_cases)):
            if cfg.data.test.test_cases[i].type in ['CityscapesDataset', 'ACDCDataset','KITTIDataset','NightCityDataset']:
                # hard code index
                cfg.data.test.test_cases[i].pipeline[1].img_ratios = [
                    0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
                cfg.data.test.test_cases[i].pipeline[1].flip = True

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

    cfg.model.pretrained = None
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
    cfg.model.backbone.prompt_sparse_rate = args.prompt_sparse_rate
    #cfg.model.class_names=datasets[0].CLASSES
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

    file_name=args.method
    model_name='segformer' if 'segformer' in args.config else ('segnext' if 'segnext' in args.config else '')
    dataset_name='acdc' if 'acdc' in args.config else ('night' if 'night' in args.config  else '')
    print(file_name+ '_' + model_name+ '_'+ dataset_name)

    #checkpoint = load_checkpoint(model, cfg.model.pretrained, map_location='cpu')
    # model.CLASSES = checkpoint['meta']['CLASSES']
    # model.PALETTE = checkpoint['meta']['PALETTE']
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
    ema_model = create_ema_model(model) #?

    frame_passed=0
    # tuning on continual step
    cnt = 0
    num_itr = 3
    All_mIoU = 0
    total_predict_time=0
    total_processed_frame=0
    for i in range(10):
        mean_mIoU = 0
        print("revisit times:",i)
        j=0
        for dataset, data_loader in zip(datasets, data_loaders):
            j=j+1
            pred_begin = time.time()
            outputs = single_gpu_svdp(args, model, data_loader, args.show, args.show_dir,
                                      efficient_test, anchor, ema_model, anchor_model, False, False)

            # outputs,frame_passed,domains_detections = single_gpu_ours(model, data_loader, args.show, args.show_dir,
            #                               efficient_test,anchor, ema_model, anchor_model,frame_passed, domains_detections,i*4+j)

            total_predict_time = total_predict_time+time.time()-pred_begin
            total_processed_frame=total_processed_frame+len(data_loader)

            rank, _ = get_dist_info()
            if rank == 0:
                if args.out:
                    print(f'\nwriting results to {args.out}')
                    mmcv.dump(outputs, args.out)
                kwargs = {} if args.eval_options is None else args.eval_options
                if args.format_only:
                    dataset.format_results(outputs, **kwargs)
                if args.eval:
                    results = dataset.evaluate(outputs, args.eval, **kwargs)
                    # mIoU = results['mIoU']
                    # wandb.log(
                    #     {
                    #         "mIoU": mIoU,
                    #     }
                    # )
                    # print('1')
                    # mean_mIoU += mIoU
    #
    #     wandb.log(
    #         {
    #             "mean_mIoU": mean_mIoU / 4,
    #         }
    #     )
    #     All_mIoU = All_mIoU + mean_mIoU / 4
    # wandb.log(
    #     {
    #         "All_mIoU": All_mIoU / num_itr,
    #     }
    # )
    # wandb.finish()

    print("total avg pred time:%.3f seconds; " % (total_predict_time / total_processed_frame))


if __name__ == '__main__':
    main()
