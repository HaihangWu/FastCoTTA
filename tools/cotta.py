import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import single_model_update, single_gpu_cotta,Efficient_adaptation,DPT,single_gpu_ours,single_gpu_AuxAdapt
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from IPython import embed
from copy import deepcopy
import time
from collections import deque


def create_ema_model(model):
    ema_model = deepcopy(model)#get_model(args.model)(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
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
    parser.add_argument(
        '--method',
        choices=['Source', 'BN', 'TENT', 'AuxAdapt', 'DPT', 'ETA', 'CoTTA', 'Ours'],
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
    if 'AuxAdapt' in args.method:
        config_s= args.config.replace("base", "tiny") if 'segnext' in args.config else args.config.replace("b5", "b0")
        cfg_s = mmcv.Config.fromfile(config_s)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
        if 'AuxAdapt' in args.method:
            cfg_s.merge_from_dict(args.options)
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
                if 'Source' in args.method or 'BN' in args.method or 'TENT' in args.method:
                    cfg.data.test.test_cases[i].pipeline[1].img_ratios = [1.0]
                    cfg.data.test.test_cases[i].pipeline[1].flip = False
                elif 'AuxAdapt' in args.method or 'Ours' in args.method:
                    cfg.data.test.test_cases[i].pipeline[1].img_ratios = [1.0, 2.0]
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

    file_name=args.method
    file_name =file_name+ '_' + 'segformer' if 'segformer' in args.config else ('segnext' if 'segnext' else '')
    file_name =file_name+ '_'+ 'acdc' if 'acdc' in args.config else ('night' if 'night' else '')
    print(file_name)

    if 'TENT' in args.method:
        for name, param in model.named_parameters():
            if ("norm" in name or "bn" in name or "ln" in name or "BatchNorm" in name):
                    param.requires_grad = True
            else:
                param.requires_grad = False
    #checkpoint = load_checkpoint(model, cfg.model.pretrained, map_location='cpu')
    # model.CLASSES = checkpoint['meta']['CLASSES']
    # model.PALETTE = checkpoint['meta']['PALETTE']
    pretrained_dict = torch.load(cfg.model.pretrained,map_location='cpu')
    #print(pretrained_dict['state_dict'].keys())
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
    if 'AuxAdapt' in args.method:
        cfg_s.model.train_cfg = None
        model_s = build_segmentor(cfg_s.model, test_cfg=cfg_s.get('test_cfg'))
        pretrained_dict = torch.load(cfg_s.model.pretrained, map_location='cpu')
        model_s.load_state_dict(pretrained_dict['state_dict'])
        model_s.CLASSES = datasets[0].CLASSES
        model_s.PALETTE = datasets[0].PALETTE
        model_s = MMDataParallel(model_s, device_ids=[0])

    anchor = deepcopy(model.state_dict()) #?
    anchor_model = deepcopy(model) #?
    ema_model = create_ema_model(model) #?
    for name, param in anchor_model.named_parameters():
        if "DSP" in name or "DAP" in name:
                param = torch.zeros_like(param)
    print([param.data for name, param in anchor_model.named_parameters() if "DSP" in name or "DAP" in name])
    frame_passed=0
    total_predict_time=0
    domains_detections={}
    # domains_detections["detection"]=True
    # domains_detections["detection_prob"]=0.1
    # domains_detections["adaptation_prob"] = {}
    # domains_detections["cur_adaptation_prob"] = 1.0
    # domains_detections["cur_dom"]="domain"+str(1)
    # #domains_detections["shift"]=False
    # domains_detections["storage"] = []
    # domains_detections["adapted_frame"] = 0

    # domains_detections["hp_k"] = 20
    # domains_detections["hp_reso_select_k"] = 5
    # domains_detections["hp_z_dm_shift"] = 3.0
    # domains_detections["hp_z_adapt_ends"] = -0.5
    # domains_detections["dm_shift"] = True
    # domains_detections["dm_reso_select_processed_frames"] = -1
    # domains_detections["dm_reso_select_conf_info"]=[[],[]]
    # domains_detections["adaptation"] = True
    # domains_detections["pred_conf"] = [[],[]] #deque(maxlen=(2*domains_detections["hp_k"]))
    # domains_detections["domain_conf"]=[]
    domains_detections["conf_gain"] = [0]
    domains_detections["adat_ends"]=0.005
    domains_detections["hp_k"] = 20
    domains_detections["imge_id"] = 0


    total_predict_time=0
    total_processed_frame=0
    current_model_probs=None
    for i in range(10):
        print("revisit times:",i)
        j=0
        for dataset, data_loader in zip(datasets, data_loaders):
            j=j+1
            pred_begin = time.time()
            if 'Source' in args.method or 'BN' in args.method or 'TENT' in args.method:
                outputs = single_model_update(model, data_loader, args, efficient_test)
            elif 'AuxAdapt'in args.method:
                outputs=single_gpu_AuxAdapt(model,model_s,data_loader,efficient_test,frame_passed)
            elif 'VDP'in args.method:
                outputs, frame_passed, ldelta = DPT(model, data_loader, ldelta,
                                                    efficient_test, ema_model, anchor_model, frame_passed, i * 4 + j)
            elif 'ETA'in args.method:
                outputs,frame_passed = Efficient_adaptation(model, data_loader, current_model_probs,
                                          efficient_test,anchor, ema_model, anchor_model,frame_passed, i*4+j)
            elif 'CoTTA'in args.method:
                outputs,frame_passed = single_gpu_cotta(model, data_loader, args.show, args.show_dir,
                                          efficient_test,anchor, ema_model, anchor_model,frame_passed, i*4+j)
            elif 'Ours'in args.method:
                outputs,frame_passed,domains_detections = single_gpu_ours(model, data_loader, args.show, args.show_dir,
                                          efficient_test,anchor, ema_model, anchor_model,frame_passed, domains_detections,i*4+j)

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
                    dataset.evaluate(outputs, args.eval, **kwargs)
    print("total avg pred time:%.3f seconds; " % (total_predict_time / total_processed_frame))

if __name__ == '__main__':
    main()
