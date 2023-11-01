import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from torch.distributions import Categorical
import torch.nn.functional as F
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from IPython import embed
from mmseg.ops import resize
import time
import random
from copy import deepcopy
from scipy.stats import wasserstein_distance
from scipy.special import expit
import copy
import math


def update_ema_variables(ema_model, model, alpha_teacher, iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    if True:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

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


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False
                    ):
    #model.eval()
    model.train() # for BN
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    pred_begin=time.time()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result_ori, probs, preds = model(return_loss=False, **data)
            result = [preds[0][0].astype(np.int64)]

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)
        # batch_size = data['img'][0].size(0)
        # if i==399: # hide progress
        #     for _ in range(batch_size):
        #         prog_bar.update()
    pred_time = time.time() - pred_begin
    print("pred_time: %.3f seconds;" % (pred_time / (i + 1)))
    return results


def single_gpu_cotta(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    anchor=None,
                    ema_model=None,
                    anchor_model=None,
                     frame_passed =0,
                     round=-1):
    """Test with single GPU.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    anchor_model.eval()
    results = []
    dataset = data_loader.dataset
    # prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []
    out_dir = "./Cotta/"+str(frame_passed)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append(param)
            #print(name)
        else:
            param.requires_grad=False
    optimizer = torch.optim.Adam(param_list, lr=0.00006/8, betas=(0.9, 0.999))# for segformer
    #optimizer = torch.optim.SGD(param_list, lr=0.01 / 8)  # for SETR
    pred_time=0
    for i, data in enumerate(data_loader):
        model.eval() # student model
        ema_model.eval() # teacher model
        anchor_model.eval() # source model
        #pred_begin=time.time()
        # if i==0:
        #     ema_model.load_state_dict(anchor)
        frame_passed=frame_passed +1
        with torch.no_grad():
            img_id = 0
            if len(data['img']) == 14:
                img_id = 4  # The default size without flip
            result, probs_, preds_ = anchor_model(return_loss=False, img=[data['img'][img_id]],img_metas=[data['img_metas'][img_id].data[0]])#**data)
            mask = (torch.amax(probs_[0], 0).cpu().numpy() > 0.69).astype(np.int64)
            result, probs, preds = ema_model(return_loss=False, **data)

            result = [(mask*preds[img_id][0] + (1.-mask)*result[0]).astype(np.int64)]

            #result = [(mask * preds[0][0] + (1. - mask) * preds[1][0]).astype(np.int64)]
            # result_H, probs_H, preds_H = anchor_model(return_loss=False, img=[data['img'][1]],
            #                                       img_metas=[data['img_metas'][1].data[0]])
            # result_L, probs_L, preds_L = anchor_model(return_loss=False, img=[data['img'][img_id]],
            #                                       img_metas=[data['img_metas'][img_id].data[0]])
            # result = [(mask * result_L[0] + (1. - mask) * result_H[0]).astype(np.int64)]

            weight = 1.
        # if (show or out_dir) and (round ==0 or round==4 or round==9):
        #     img_tensor = data['img'][0]
        #     img_metas = data['img_metas'][0].data[0]
        #     imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        #     assert len(imgs) == len(img_metas)
        #     for img, img_meta in zip(imgs, img_metas):
        #         h, w, _ = img_meta['img_shape']
        #         img_show = img[:h, :w, :]
        #
        #         ori_h, ori_w = img_meta['ori_shape'][:-1]
        #         img_show = mmcv.imresize(img_show, (ori_w, ori_h))
        #
        #         if out_dir:
        #             out_file = osp.join(out_dir, img_meta['ori_filename'])
        #         else:
        #             out_file = None
        #
        #         model.module.show_result(
        #             img_show,
        #             result,
        #             palette=dataset.PALETTE,
        #             show=show,
        #             out_file=out_file)
        if isinstance(result, list):
            if len(data['img'])==14:
                img_id = 4 #The default size without flip
            else:
                img_id = 0
            #student_begin = time.time()
            loss = model.forward(return_loss=True, img=data['img'][img_id], img_metas=data['img_metas'][img_id].data[0], gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0))
            #student_pred = time.time() - student_begin
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        torch.mean(weight*loss["decode.loss_seg"]).backward()
        optimizer.step()
        optimizer.zero_grad()

        ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=0.999) #teacher model

        #stochastic restoration
        for nm, m  in model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape)<0.01).float().cuda()
                    with torch.no_grad():
                        p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)


        #pred_time += time.time() - pred_begin
        # batch_size = data['img'][0].size(0)
        # if i==399:
        #     for _ in range(batch_size):
        #         prog_bar.update()
        #print("iter %d, teacher_pred: %.3f seconds; anchor_pred: %.3f;" % (i, teacher_pred, anchor_pred))
        #print("iter %d, teacher_pred: %.3f seconds; student_pred: %.3f; student_train: %.3f;model_update_time: %.3f;restoration_time: %.3f;" % (i,teacher_pred,student_pred,student_train,model_update_time,restoration_time))
    #print("pred_time: %.3f seconds;" % (pred_time/(i+1)))
    return results,frame_passed

def Efficient_adaptation(model,
                    data_loader,
                    current_model_probs,
                    efficient_test=False,
                    anchor=None,
                    ema_model=None,
                    anchor_model=None,
                     frame_passed =0,
                     round=-1):
    """Test with single GPU.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    anchor_model.eval()
    results = []
    dataset = data_loader.dataset
    # prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []
    out_dir = "./Cotta/"+str(frame_passed)
    E0=torch.tensor(1.8*math.log(19.0))
    redundancy_epson=0.1
    back_img_count=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append(param)
            #print(name)
        else:
            param.requires_grad=False
    optimizer = torch.optim.Adam(param_list, lr=0.00006/8, betas=(0.9, 0.999))# for segformer
    #optimizer = torch.optim.SGD(param_list, lr=0.01 / 8)  # for SETR
    pred_time=0
    for i, data in enumerate(data_loader):
        model.eval() # student model
        ema_model.eval() # teacher model
        anchor_model.eval() # source model
        #pred_begin=time.time()
        # if i==0:
        #     ema_model.load_state_dict(anchor)
        frame_passed=frame_passed +1
        with torch.no_grad():
            img_id = 0
            if len(data['img']) == 14:
                img_id = 4 # The default size without flip
            result, probs_, preds_ = anchor_model(return_loss=False, img=[data['img'][img_id]],img_metas=[data['img_metas'][img_id].data[0]])#**data)
            entropy_pred=torch.mean(Categorical(probs = probs_.view(-1, probs_.shape[-1])).entropy())
            # if current_model_probs is None:
            #     current_model_probs=copy.deepcopy(probs_.view(-1, probs_.shape[-1]).mean(0))
            #     cosine_similarities = torch.tensor(1.0)
            # else:
            #     cosine_similarities = F.cosine_similarity(current_model_probs,probs_.view(-1, probs_.shape[-1]).mean(0),0)
            #     print("redundant sample", i, cosine_similarities, redundancy_epson,current_model_probs,probs_.view(-1, probs_.shape[-1]).mean(0))
            #     current_model_probs=copy.deepcopy(0.9 * current_model_probs + (1 - 0.9) * probs_.view(-1, probs_.shape[-1]).mean(0))
            # if torch.abs(cosine_similarities) > redundancy_epson:
            #     continue

            mask = (torch.amax(probs_[0], 0).cpu().numpy() > 0.69).astype(np.int64)
            result, probs, preds = ema_model(return_loss=False, **data)

            result = [(mask*preds[img_id][0] + (1.-mask)*result[0]).astype(np.int64)]

            #result = [(mask * preds[0][0] + (1. - mask) * preds[1][0]).astype(np.int64)]
            # result_H, probs_H, preds_H = anchor_model(return_loss=False, img=[data['img'][1]],
            #                                       img_metas=[data['img_metas'][1].data[0]])
            # result_L, probs_L, preds_L = anchor_model(return_loss=False, img=[data['img'][img_id]],
            #                                       img_metas=[data['img_metas'][img_id].data[0]])
            # result = [(mask * result_L[0] + (1. - mask) * result_H[0]).astype(np.int64)]

            weight = torch.exp(E0 - entropy_pred)
        if isinstance(result, list):
            if len(data['img'])==14:
                img_id = 4 #The default size without flip
            else:
                img_id = 0
            #student_begin = time.time()
            if entropy_pred<E0:
                back_img_count = back_img_count + 1
                loss = model.forward(return_loss=True, img=data['img'][img_id], img_metas=data['img_metas'][img_id].data[0], gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0))
            #student_pred = time.time() - student_begin
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        if entropy_pred<E0:
            torch.mean(weight*loss["decode.loss_seg"]).backward()
            optimizer.step()
            optimizer.zero_grad()

            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=0.999) #teacher model

            #stochastic restoration
            for nm, m  in model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<0.01).float().cuda()
                        with torch.no_grad():
                            p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)


        #pred_time += time.time() - pred_begin
        # batch_size = data['img'][0].size(0)
        # if i==399:
        #     for _ in range(batch_size):
        #         prog_bar.update()
        #print("iter %d, teacher_pred: %.3f seconds; anchor_pred: %.3f;" % (i, teacher_pred, anchor_pred))
        #print("iter %d, teacher_pred: %.3f seconds; student_pred: %.3f; student_train: %.3f;model_update_time: %.3f;restoration_time: %.3f;" % (i,teacher_pred,student_pred,student_train,model_update_time,restoration_time))
    #print("pred_time: %.3f seconds;" % (pred_time/(i+1)))
    print("reliable samples", back_img_count)
    return results,frame_passed






