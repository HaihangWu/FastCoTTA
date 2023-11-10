import os.path as osp
import pickle
import shutil
import tempfile
import datetime

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

def single_gpu_ours(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    anchor=None,
                    ema_model=None,
                    anchor_model=None,
                     frame_passed =0,
                    domains_detections={},
                     round=-1):
    model.eval()
    anchor_model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []
    out_dir = "./Cotta/"+str(frame_passed)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append(param)
        else:
            param.requires_grad=False
    optimizer = torch.optim.Adam(param_list, lr=0.00006, betas=(0.9, 0.999))# for segformer, segnext
    pred_begin=time.time()
    teacher_pred_conf=[]
    print("new domain starts,",frame_passed)
    for i, data in enumerate(data_loader):
        model.eval() # student model
        ema_model.eval() # teacher model
        anchor_model.eval() # source model
        frame_passed=frame_passed +1
        ######### Domain resolution selector/Adaptation trigger##################
        # if domains_detections["dm_reso_select_processed_frames"] > domains_detections["hp_reso_select_k"]:
        #     if np.mean(domains_detections["dm_reso_select_conf_info"][0]) > np.mean(
        #             domains_detections["dm_reso_select_conf_info"][1]):
        #         domains_detections["imge_id"] = 0
        #     else:
        #         domains_detections["imge_id"] = 1
        #     domains_detections["dm_reso_select_processed_frames"] = -1
        #     # domains_detections["adaptation"] = True
        #     print("domain resolution selection", domains_detections["imge_id"], np.mean(domains_detections["dm_reso_select_conf_info"][0]) ,np.mean(domains_detections["dm_reso_select_conf_info"][1]) ,round, frame_passed)

        # if domains_detections["dm_shift"]:
        #     domains_detections["dm_reso_select_processed_frames"] = 0
        #     domains_detections["pred_conf"]=[[],[]]
        #     domains_detections["adaptation"] = False
        #     domains_detections["dm_shift"]=False

        ### dynamic adaptation: we make assumption that frames in the same domain are similar
        # if np.sum([len(sublist) for sublist in domains_detections["pred_conf"]]) >= (domains_detections["hp_k"]-domains_detections["hp_reso_select_k"]):
        #     # imge_id = 0
        #     # if domains_detections["adaptation"]:
        #     #     imge_id = domains_detections["imge_id"]
        #     source_pred_mean_s = np.mean(domains_detections["dm_reso_select_conf_info"][0])
        #     source_pred_std_s = np.std(domains_detections["dm_reso_select_conf_info"][0])
        #     source_pred_mean_l = np.mean(domains_detections["dm_reso_select_conf_info"][1])
        #     source_pred_std_l = np.std(domains_detections["dm_reso_select_conf_info"][1])
        #     TS_distance_s=0
        #     TS_distance_l=0
        #     if len(domains_detections["pred_conf"][0])>1.5:
        #         teacher_pred_mean_s=np.mean(domains_detections["pred_conf"][0])
        #         teacher_pred_std_s = np.mean(domains_detections["pred_conf"][0])
        #         TS_distance_s = (teacher_pred_mean_s - source_pred_mean_s) / np.sqrt(
        #             source_pred_std_s ** 2.0 + teacher_pred_std_s ** 2.0)
        #     if len(domains_detections["pred_conf"][1])>1.5:
        #         teacher_pred_mean_l=np.mean(domains_detections["pred_conf"][1])
        #         teacher_pred_std_l = np.mean(domains_detections["pred_conf"][1])
        #         TS_distance_l = (teacher_pred_mean_l - source_pred_mean_l) / np.sqrt(
        #             source_pred_std_l ** 2.0 + teacher_pred_std_l ** 2.0)
        #     TS_distance=TS_distance_s*(len(domains_detections["pred_conf"][0])/np.sum([len(sublist) for sublist in domains_detections["pred_conf"]]))\
        #                 +TS_distance_l*(len(domains_detections["pred_conf"][1])/np.sum([len(sublist) for sublist in domains_detections["pred_conf"]]))
        #     if TS_distance<domains_detections["hp_z_adapt_ends"] and domains_detections["adaptation"]:
        #         domains_detections["adaptation"] = False
        #     if TS_distance > domains_detections["hp_z_adapt_ends"] and not domains_detections["adaptation"]:
        #         domains_detections["adaptation"] = True
        #     print("adaptation  test:", domains_detections["adaptation"], TS_distance,TS_distance_s,TS_distance_l, np.sum([len(sublist) for sublist in domains_detections["pred_conf"]]), frame_passed, round)

        with torch.no_grad():
                ######### domain shift detection##################
                # if len(domains_detections["pred_conf"])>= (2*domains_detections["hp_k"]):
            # domain_shift_detection=True if (frame_passed % domains_detections["hp_k"] == 0
            #                                 or ((frame_passed+1) % domains_detections["hp_k"] == 0 and len(domains_detections["domain_conf"])<1.5)) else False
            # if domain_shift_detection:
            #     imge_id = 0
            #     result, probs, preds = anchor_model(return_loss=False, img=[data['img'][imge_id]],
            #                                      img_metas=[data['img_metas'][imge_id].data[0]])
            #     domain_distance=0.0
            #     if len(domains_detections["domain_conf"])>1.5:
            #         first_domain_mean = np.mean(domains_detections["domain_conf"])
            #         first_domain_std = np.std(domains_detections["domain_conf"])
            #         second_domain_mean = np.mean(torch.amax(probs[0], 0).cpu().numpy())
            #         domain_distance = abs(first_domain_mean - second_domain_mean) / np.sqrt(
            #             first_domain_std ** 2.0)
            #         print("domain shift test", domain_distance, first_domain_mean, second_domain_mean,  len(domains_detections["domain_conf"]), frame_passed, round)
            #     if domain_distance > domains_detections["hp_z_dm_shift"]:
            #         domains_detections["dm_shift"] = True
            #         domains_detections["domain_conf"] = []
            #     else:
            #         domains_detections["domain_conf"].append(np.mean(torch.amax(probs[0], 0).cpu().numpy()))

            ######### domain resolution selector##################
            # if domains_detections["dm_reso_select_processed_frames"]>=0 and not domain_shift_detection: #and domains_detections["dm_reso_select_processed_frames"] < domains_detections["hp_k"]:
            #     result, probs, preds = anchor_model(return_loss=False, **data)
            #     domains_detections["dm_reso_select_conf_info"][0].append(np.mean(probs[0])) # low reso conf
            #     domains_detections["dm_reso_select_conf_info"][1].append(np.mean(probs[1])) # high reso conf
            #     result = [(result[-1]).astype(np.int64)]
            #     domains_detections["dm_reso_select_processed_frames"]=domains_detections["dm_reso_select_processed_frames"]+1

            #########  after selecting the domain resolution, use teacher model for prediction
            # if domains_detections["dm_reso_select_processed_frames"] < 0 and not domain_shift_detection:
            #     imge_id=0
            #     if domains_detections["adaptation"]:
            #         imge_id = domains_detections["imge_id"]
            #     result, probs, preds = ema_model(return_loss=False, img=[data['img'][imge_id]],img_metas=[data['img_metas'][imge_id].data[0]])
            #     if imge_id==0:
            #         domains_detections["pred_conf"][0].append(np.mean(torch.amax(probs[0], 0).cpu().numpy()))
            #     else:
            #         domains_detections["pred_conf"][1].append(np.mean(torch.amax(probs[0], 0).cpu().numpy()))

                if np.mean(domains_detections["conf_gain"])<domains_detections["adat_ends"]:
                    result, probs, preds = ema_model(return_loss=False, img=[data['img'][domains_detections["imge_id"]]],
                                                     img_metas=[data['img_metas'][domains_detections["imge_id"]].data[0]])
                    result_source, probs_source, preds_source = anchor_model(return_loss=False, img=[data['img'][domains_detections["imge_id"]]],
                                                     img_metas=[data['img_metas'][domains_detections["imge_id"]].data[0]])
                    domains_detections["conf_gain"].append(np.mean(torch.amax(probs[0], 0).cpu().numpy())-np.mean(torch.amax(probs_source[0], 0).cpu().numpy()))
                else:
                    result, probs, preds = ema_model(return_loss=False, img=[data['img'][domains_detections["imge_id"]]],
                                                     img_metas=[data['img_metas'][domains_detections["imge_id"]].data[0]])
        if isinstance(result, list):
            if len(data['img']) == 14:
                img_id = 4  # The default size without flip
            else:
                img_id = 0
            if np.mean(domains_detections["conf_gain"])<domains_detections["adat_ends"]:
                loss = model.forward(return_loss=True, img=data['img'][img_id], img_metas=data['img_metas'][img_id].data[0],
                                     gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0))
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        if np.mean(domains_detections["conf_gain"])<domains_detections["adat_ends"]:
            torch.mean(loss["decode.loss_seg"]).backward()
            optimizer.step()
            optimizer.zero_grad()
            ema_model = update_ema_variables(ema_model=ema_model, model=model, alpha_teacher=0.999)  # teacher model

    pred_time = time.time() - pred_begin
    print("average pred_time: %.3f seconds; average conf gain: %.3f" % (pred_time/(i+1),np.mean(domains_detections["adaptation"])))
    return results,frame_passed,domains_detections

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

def DPT(model,
        data_loader,
        ldelta=None,
        efficient_test=False,
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
    student_DAP_old_domain=None
    student_DAP=None
    sdelta=None
    alph=0.01
    for name, param in model.named_parameters():
        if "DSP" in name or "DAP" in name:
            param_list.append(param)
            if "DAP" in name:
                student_DAP_old_domain=copy.deepcopy(param.data)
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
            mask = (torch.amax(probs_[0], 0).cpu().numpy() > 0.69).astype(np.int64)
            result, probs, preds = ema_model(return_loss=False, **data)

            result = [(mask*preds[img_id][0] + (1.-mask)*result[0]).astype(np.int64)]

            weight = 1.0
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

        DAP_loss = 0
        if round>0.5 and student_DAP is not None:
            DAP_loss=torch.mean(alph*ldelta*(student_DAP-student_DAP_old_domain)**2)
        Total_loss=torch.mean(weight * loss["decode.loss_seg"])+DAP_loss
        Total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for [ema_name, ema_param], [name, param] in zip(ema_model.named_parameters(), model.named_parameters()):
            if ("DSP" in name and "DSP" in ema_name) or ("DAP" in name and "DAP" in ema_name):
                ema_param.data[:] = 0.999 * ema_param[:].data[:] + (1 - 0.999) * param[:].data[:]
                print(name)
            if "DAP" in name:
                if round>0.5:
                    if i==0:
                        yita=(param.data-student_DAP_old_domain)*param.grad
                    if i > 1:
                        ldelta = ldelta - yita / (sdelta + 0.01)
                    if i > 0:
                        sdelta=(param.data-student_DAP_old_data)**2
                        ldelta=ldelta+yita/(sdelta+0.01)
                    student_DAP_old_data = copy.deepcopy(param.data)
                    student_DAP=param
                else:
                    ldelta=param.data-param.data

        #ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=0.999) #teacher model

        # #stochastic restoration
        # for nm, m  in model.named_modules():
        #     for npp, p in m.named_parameters():
        #         if npp in ['weight', 'bias'] and p.requires_grad:
        #             mask = (torch.rand(p.shape)<0.01).float().cuda()
        #             with torch.no_grad():
        #                 p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)
    return results,frame_passed,ldelta

def single_gpu_AuxAdapt(model_l,
                        model_s,
                    data_loader,
                    efficient_test=False,
                    frame_passed=0):
    model_l.eval()
    model_s.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []
    out_dir = "./cotta/" + str(datetime.datetime.now())
    for name, param in model_s.named_parameters():
        if param.requires_grad:
            param_list.append(param)
            # print(name)
        else:
            param.requires_grad = False
    optimizer = torch.optim.Adam(param_list, lr=0.00006 / 8, betas=(
    0.9, 0.999))  # for segformer; 0.00006/8 is not large enough to learn target domain quickly
    # optimizer = torch.optim.SGD(param_list, lr=0.01/8 )  # for SETR;
    pred_time = 0
    for i, data in enumerate(data_loader):
        pred_begin = time.time()

        with torch.no_grad():
            img_id = 0
            # result, probs, preds = ema_model(return_loss=False, **data)
            # mask = (probs[0][0] > probs[1][0]).astype(np.int64)
            # result = [(mask * preds[0][0] + (1. - mask) * preds[1][0]).astype(np.int64)]
            result_L, probs_L, preds_L = model_l(return_loss=False, img=[data['img'][1]],
                                                 img_metas=[data['img_metas'][1].data[0]])
            result_s, probs_s, preds_s = model_s(return_loss=False, img=[data['img'][0]],
                                                 img_metas=[data['img_metas'][0].data[0]])
            result = list(((probs_L + probs_s) / 2.0).argmax(dim=1).cpu().numpy())
            weight = 1.

        if isinstance(result, list):
            img_id = 0
            # student_begin = time.time()
            loss = model_s.forward(return_loss=True, img=data['img'][img_id],
                                   img_metas=data['img_metas'][img_id].data[0],
                                   gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0))
            # student_pred = time.time() - student_begin
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        torch.mean(weight * loss["decode.loss_seg"]).backward()
        optimizer.step()
        optimizer.zero_grad()

        # ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=0.999) #teacher model

        pred_time += time.time() - pred_begin
        # batch_size = data['img'][0].size(0)
        # if i==399: # whh
        #     for _ in range(batch_size):
        #         prog_bar.update()
        # print("iter %d, teacher_pred: %.3f seconds; anchor_pred: %.3f;" % (i, teacher_pred, anchor_pred))
        # print("iter %d, teacher_pred: %.3f seconds; student_pred: %.3f; student_train: %.3f;model_update_time: %.3f;restoration_time: %.3f;" % (i,teacher_pred,student_pred,student_train,model_update_time,restoration_time))
    print("pred_time: %.3f seconds;" % (pred_time / (i + 1)))
    return results


def single_model_update(model,
                    data_loader,
                        args,
                    efficient_test=False):
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

    if "Source" in args.method or 'TENT' in args.method:
        model.eval()
    if 'BN' in args.method:
        model.train()

    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []
    pred_conf=[]
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append(param)
    optimizer = torch.optim.Adam(param_list, lr=0.00006/8, betas=(0.9, 0.999)) # for segformer,segnext
    pred_time=0
    for i, data in enumerate(data_loader):
        pred_begin=time.time()
        with torch.no_grad():
            result, probs, preds = model(return_loss=False, **data)
            pred_conf.append(np.mean(torch.amax(probs[0], 0).cpu().numpy()))

        img_id = 0
        if isinstance(result, list):
            if "TENT" in args.method:
                loss = model.forward(return_loss=True, img=data['img'][img_id], img_metas=data['img_metas'][img_id].data[0], gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0))
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if "TENT" in args.method:
                loss = model(return_loss=True, img=data['img'][img_id], img_metas=data['img_metas'][img_id].data[0], gt_semantic_seg=result)
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        if "TENT" in args.method:
            torch.mean(loss["decode.loss_seg"]).backward()
            optimizer.step()
            optimizer.zero_grad()

        pred_time += time.time() - pred_begin
        batch_size = data['img'][0].size(0)
        if i==399:
            for _ in range(batch_size):
                prog_bar.update()
    print("pred_time: %.3f seconds; confidence: %.3f " % (pred_time/(i+1),np.mean(pred_conf)))
    return results