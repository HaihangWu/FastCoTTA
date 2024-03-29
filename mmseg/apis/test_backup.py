import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from IPython import embed
from mmseg.ops import resize
import time
import random
from copy import deepcopy
from scipy.stats import wasserstein_distance

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
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    pred_time=0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result_ori, probs, preds = model(return_loss=False, **data)
            result = [preds[0][0].astype(np.int64)]

        # if show or out_dir:
        #     img_tensor = data['img'][0]
        #     img_metas = data['img_metas'][0].data[0]
        #     imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        #     assert len(imgs) == len(img_metas)
        #
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
    #optimizer = torch.optim.Adam(param_list, lr=0.00006/8, betas=(0.9, 0.999))# for segformer
    optimizer = torch.optim.SGD(param_list, lr=0.01 / 8)  # for SETR
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


def single_gpu_language_cotta(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    anchor=None,
                    ema_model=None,
                    anchor_model=None,
                     frame_passed =0,
                    domains_detections={},
                    model_name="setr",
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
    domain_storage_length=100
    #storage_temp_length=10
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []
    out_dir = "./Cotta/"+str(frame_passed)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append(param)
        else:
            param.requires_grad=False
    if model_name in "setr":
        optimizer = torch.optim.SGD(param_list, lr=0.01)  # for SETR
        print("setr model")
    else:
        optimizer = torch.optim.Adam(param_list, lr=0.00006, betas=(0.9, 0.999))# for segformer

    pred_time=0
    print("new domain starts,",frame_passed)
    new_domain_frame=frame_passed
    for i, data in enumerate(data_loader):
        model.eval() # student model
        ema_model.eval() # teacher model
        anchor_model.eval() # source model
        pred_begin=time.time()
        # if i==0:
        #     ema_model.load_state_dict(anchor)
        #if frame_passed%100==0
        # if random.random()<domains_detections["detection_prob"]:
        #     domains_detections["detection"] = True
        # else:
        #     domains_detections["detection"] = False
        frame_passed=frame_passed +1
        with torch.no_grad():
            img_id = 0
            if len(data['img']) == 14:
                img_id = 4  # The default size without flip

            if domains_detections["get_new_domain_info"]:
                result, probs_, preds_ = anchor_model(return_loss=False, img=[data['img'][img_id]],img_metas=[data['img_metas'][img_id].data[0]])
                domains_detections["get_conf_by_source"].append(np.mean(torch.amax(probs_[0], 0).cpu().numpy()))
            if len(domains_detections["get_conf_by_source"])>=domains_detections["info_length_by_source"]:
                cur_domain_mean=np.mean(domains_detections["get_conf_by_source"])
                cur_domain_std = np.std(domains_detections["get_conf_by_source"])
                z_score = 10000
                domain_index=-1
                for k,v in domains_detections["domain_info"].items():
                    this_domain_mean=np.mean(v[0])
                    this_domain_std=np.std(v[0])
                    z_score_temp = abs(cur_domain_mean - this_domain_mean) / np.sqrt(cur_domain_std ** 2.0 + this_domain_std ** 2.0)
                    if z_score_temp<domains_detections["z_score_threshold"] and z_score>z_score_temp:
                        z_score=z_score_temp
                        domain_index=k
                if domain_index>0.5:
                    domains_detections["ini_wass_dist"]=domains_detections["domain_info"][domain_index][1]
                    if domains_detections["cur_dom"] != domain_index:
                        domains_detections["domain_info"][domain_index][2]=domains_detections["domain_info"][domain_index][2]+1
                    domains_detections["created_new_domain"] = False
                    domains_detections["cur_dom"] = domain_index
                    domains_detections["adapt_termination_param"]=expit(domains_detections["domain_info"][domain_index][2])
                    print("revisit domain",z_score, domain_index,domains_detections["adapt_termination_param"])
                else:
                    new_domain_index=max([ k for k in domains_detections["domain_info"].keys()]+[0])+1
                    domains_detections["domain_info"][new_domain_index] = [[],[],0]
                    domains_detections["domain_info"][new_domain_index][0]=copy.deepcopy(domains_detections["get_conf_by_source"])
                    domains_detections["cur_dom"] = new_domain_index
                    domains_detections["created_new_domain"] = True
                    domains_detections["adapt_termination_param"] = expit(0)
                domains_detections["get_new_domain_info"]=False
                domains_detections["get_conf_by_source"]=[]

            if len(domains_detections["storage"])>domains_detections["storage_length"] and not domains_detections["get_new_domain_info"]:
                if not domains_detections["created_new_domain"]:
                       cur_distribution=np.array(copy.deepcopy(domains_detections["storage"][:domains_detections["storage_length"]]))
                       cur_sample=domains_detections["storage"][-1]
                       if (cur_sample < np.percentile(cur_distribution, 5) or cur_sample > np.percentile(cur_distribution, 95)):
                           domains_detections["outlier_count"] = domains_detections["outlier_count"] + 1
                       else:
                           if domains_detections["outlier_count"] >0.5:
                              domains_detections["outlier_count"]=0
                       if domains_detections["outlier_count"]>=domains_detections["outlier_threshold"]:
                            print("domain shift detected,",frame_passed)
                            domains_detections["get_new_domain_info"]=True
                            domains_detections["domain_shift_detected"] = True
                            domains_detections["ini_wass_dist"]=[]

            if len(domains_detections["storage"])>=(2*domains_detections["storage_length"]): #and domains_detections["detection"] is True:
                last_distribution = np.array(copy.deepcopy(domains_detections["storage"][:domains_detections["storage_length"]]))
                cur_distribution = np.array(copy.deepcopy(domains_detections["storage"][domains_detections["storage_length"]:]))
                wass_dist=wasserstein_distance(last_distribution,cur_distribution)

                #print("domain detection", last_mean, cur_mean, wass_dist,  frame_passed)
                if len(domains_detections["ini_wass_dist"])<domains_detections["wass_dist_length"]:
                    if domains_detections["created_new_domain"]:
                        domains_detections["ini_wass_dist"].append(wass_dist)
                else:
                    if domains_detections["created_new_domain"]:
                        domain_info_index=[k for k,v in domains_detections["domain_info"].items() if len(v[1])<0.5]
                        domains_detections["domain_info"][domain_info_index[0]][1]=copy.deepcopy(domains_detections["ini_wass_dist"])
                        domains_detections["created_new_domain"]=False
                        print("new domain created",domains_detections["domain_info"])
                    domains_detections["cur_wass_dist"].append(wass_dist)
                    if len(domains_detections["cur_wass_dist"])>=domains_detections["wass_dist_length"]:
                        if np.mean(domains_detections["cur_wass_dist"])>(domains_detections["adapt_termination_param"]*np.mean(domains_detections["ini_wass_dist"])) and not domains_detections["adaptation"]: #and (abs(cur_mean-last_mean)/np.sqrt(cur_distri_std**2.0+last_distri_std**2.0))>2.0:
                            if domains_detections["domain_shift_detected"]:
                                domains_detections["adaptation"] = True
                                #domains_detections["validation_frame"] = [[],[]]
                                print("domain adaptation begin",frame_passed)
                        if np.mean(domains_detections["cur_wass_dist"])<(domains_detections["adapt_termination_param"]*np.mean(domains_detections["ini_wass_dist"])) and domains_detections["adaptation"]:
                            domains_detections["adaptation"] = False
                            domains_detections["domain_shift_detected"] = False
                            #domains_detections["validation_frame"] = [[],[]]
                            print("domain adaptation termination",frame_passed)
                        domains_detections["cur_wass_dist"]=domains_detections["cur_wass_dist"][1:]

                domains_detections["storage"] = domains_detections["storage"][domains_detections["storage_length"]:] # detect every storage_temp_length frames
                domains_detections["storage"] = domains_detections["storage"][1:]


            if not domains_detections["adaptation"]:
                result_ori, probs, preds = ema_model(return_loss=False, img=[data['img'][img_id]],
                                                      img_metas=[data['img_metas'][img_id].data[0]])

                domains_detections["storage"].append(np.mean(torch.amax(probs[0], 0).cpu().numpy()))
            else:
                result_ori, probs, preds = ema_model(return_loss=False, **data)
                # print(type(probs[0]),probs[0],probs[0].size())
                # print(torch.mean(probs[0]).item())
                conf_mean=np.mean(probs[img_id])
                domains_detections["storage"].append(conf_mean)
                #domains_detections["storage"].append(np.mean(torch.amax(probs[0], 0).cpu().numpy()))


            result = [preds[img_id][0].astype(np.int64)]
            result_=[result_ori[0].astype(np.int64)]

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
        #if ((new_domain_frame+10)>frame_passed) or round<14: #
        if domains_detections["adaptation"]: #and (len(domains_detections["validation_frame"][0])==domains_detections["num_validation_frame"]):
            #model = deepcopy(ema_model)
            # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #     # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            #     param.data[:] = ema_param[:].data[:]
            if isinstance(result, list):
                if len(data['img'])==14:
                    img_id = 4 #The default size without flip
                else:
                    img_id = 0
                loss = model.forward(return_loss=True, img=data['img'][img_id], img_metas=data['img_metas'][img_id].data[0], gt_semantic_seg=torch.from_numpy(result_[0]).cuda().unsqueeze(0).unsqueeze(0))
                if efficient_test:
                    result = [np2tmp(_) for _ in result]
                results.extend(result)
            else:
                if efficient_test:
                    result = np2tmp(result)
                results.append(result)
            if domains_detections["language_regularization"]:
                torch.mean(loss["decode.loss_seg"]+loss["text_decode.loss_seg"]).backward()
            else:
                torch.mean(loss["decode.loss_seg"]).backward()
            optimizer.step()
            optimizer.zero_grad()

            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=0.999) #teacher model

            #stochastic restoration
            # for nm, m  in model.named_modules():
            # #for nm, m in ema_model.named_modules():
            #     if 'decode_head' in nm or 'backbone' in nm:
            #         for npp, p in m.named_parameters():
            #             if npp in ['weight', 'bias'] and p.requires_grad:
            #                 mask = (torch.rand(p.shape)<0.01).float().cuda()
            #                 with torch.no_grad():
            #                     p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)
        else:
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)


        pred_time += time.time() - pred_begin
    #     batch_size = data['img'][0].size(0)
    #     if i==399:
    #         for _ in range(batch_size):
    #             prog_bar.update()
    #total_predict_time=total_predict_time+pred_time
    print("average pred_time: %.3f seconds;" % (pred_time/(i+1)))
    return results,frame_passed,domains_detections



def single_gpu_language_cotta_xiao(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    anchor=None,
                    ema_model=None,
                    anchor_model=None,
                     frame_passed =0,
                    domains_detections={},
                    model_name="setr",
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
    domain_storage_length=100
    #storage_temp_length=10
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []
    out_dir = "./Cotta/"+str(frame_passed)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append(param)
        else:
            param.requires_grad=False
    if model_name in "setr":
        optimizer = torch.optim.SGD(param_list, lr=0.01)  # for SETR
        print("setr model")
    else:
        optimizer = torch.optim.Adam(param_list, lr=0.00006, betas=(0.9, 0.999))# for segformer

    pred_time=0
    print("new domain starts,",frame_passed)
    new_domain_frame=frame_passed
    for i, data in enumerate(data_loader):
        model.eval() # student model
        ema_model.eval() # teacher model
        anchor_model.eval() # source model
        pred_begin=time.time()
        # if i==0:
        #     ema_model.load_state_dict(anchor)
        #if frame_passed%100==0
        # if random.random()<domains_detections["detection_prob"]:
        #     domains_detections["detection"] = True
        # else:
        #     domains_detections["detection"] = False
        frame_passed=frame_passed +1
        with torch.no_grad():
            img_id = 0
            if len(data['img']) == 14:
                img_id = 4  # The default size without flip

            if domains_detections["get_new_domain_info"]:
                result, probs_, preds_ = anchor_model(return_loss=False, img=[data['img'][img_id]],img_metas=[data['img_metas'][img_id].data[0]])
                domains_detections["get_conf_by_source"].append(np.mean(torch.amax(probs_[0], 0).cpu().numpy()))
            if len(domains_detections["get_conf_by_source"])>=domains_detections["info_length_by_source"]:
                cur_domain_mean=np.mean(domains_detections["get_conf_by_source"])
                cur_domain_std = np.std(domains_detections["get_conf_by_source"])
                z_score = 10000
                domain_index=-1
                for k,v in domains_detections["domain_info"].items():
                    this_domain_mean=np.mean(v[0])
                    this_domain_std=np.std(v[0])
                    z_score_temp = abs(cur_domain_mean - this_domain_mean) / np.sqrt(cur_domain_std ** 2.0 + this_domain_std ** 2.0)
                    if z_score_temp<domains_detections["z_score_threshold"] and z_score>z_score_temp:
                        z_score=z_score_temp
                        domain_index=k
                if domain_index>0.5:
                    domains_detections["ini_wass_dist"]=domains_detections["domain_info"][domain_index][1]
                    if domains_detections["cur_dom"] != domain_index:
                        domains_detections["domain_info"][domain_index][2]=domains_detections["domain_info"][domain_index][2]+1
                    domains_detections["created_new_domain"] = False
                    domains_detections["cur_dom"] = domain_index
                    domains_detections["adapt_termination_param"]=expit(domains_detections["domain_info"][domain_index][2])
                    print("revisit domain",z_score, domain_index,domains_detections["adapt_termination_param"])
                else:
                    new_domain_index=max([ k for k in domains_detections["domain_info"].keys()]+[0])+1
                    domains_detections["domain_info"][new_domain_index] = [[],[],0]
                    domains_detections["domain_info"][new_domain_index][0]=copy.deepcopy(domains_detections["get_conf_by_source"])
                    domains_detections["cur_dom"] = new_domain_index
                    domains_detections["created_new_domain"] = True
                    domains_detections["adapt_termination_param"] = expit(0)
                domains_detections["get_new_domain_info"]=False
                domains_detections["get_conf_by_source"]=[]

            if len(domains_detections["storage"])>domains_detections["storage_length"] and not domains_detections["get_new_domain_info"]:
                if not domains_detections["created_new_domain"]:
                       cur_distribution=np.array(copy.deepcopy(domains_detections["storage"][:domains_detections["storage_length"]]))
                       cur_sample=domains_detections["storage"][-1]
                       if (cur_sample < np.percentile(cur_distribution, 5) or cur_sample > np.percentile(cur_distribution, 95)):
                           domains_detections["outlier_count"] = domains_detections["outlier_count"] + 1
                       else:
                           if domains_detections["outlier_count"] >0.5:
                              domains_detections["outlier_count"]=0
                       if domains_detections["outlier_count"]>=domains_detections["outlier_threshold"]:
                            print("domain shift detected,",frame_passed)
                            domains_detections["get_new_domain_info"]=True
                            domains_detections["domain_shift_detected"] = True
                            domains_detections["ini_wass_dist"]=[]

            if len(domains_detections["storage"])>=(2*domains_detections["storage_length"]): #and domains_detections["detection"] is True:
                last_distribution = np.array(copy.deepcopy(domains_detections["storage"][:domains_detections["storage_length"]]))
                cur_distribution = np.array(copy.deepcopy(domains_detections["storage"][domains_detections["storage_length"]:]))
                wass_dist=wasserstein_distance(last_distribution,cur_distribution)

                #print("domain detection", last_mean, cur_mean, wass_dist,  frame_passed)
                if len(domains_detections["ini_wass_dist"])<domains_detections["wass_dist_length"]:
                    if domains_detections["created_new_domain"]:
                        domains_detections["ini_wass_dist"].append(wass_dist)
                else:
                    if domains_detections["created_new_domain"]:
                        domain_info_index=[k for k,v in domains_detections["domain_info"].items() if len(v[1])<0.5]
                        domains_detections["domain_info"][domain_info_index[0]][1]=copy.deepcopy(domains_detections["ini_wass_dist"])
                        domains_detections["created_new_domain"]=False
                        print("new domain created",domains_detections["domain_info"])
                    domains_detections["cur_wass_dist"].append(wass_dist)
                    if len(domains_detections["cur_wass_dist"])>=domains_detections["wass_dist_length"]:
                        if np.mean(domains_detections["cur_wass_dist"])>(domains_detections["adapt_termination_param"]*np.mean(domains_detections["ini_wass_dist"])) and not domains_detections["adaptation"]: #and (abs(cur_mean-last_mean)/np.sqrt(cur_distri_std**2.0+last_distri_std**2.0))>2.0:
                            if domains_detections["domain_shift_detected"]:
                                domains_detections["adaptation"] = True
                                #domains_detections["validation_frame"] = [[],[]]
                                print("domain adaptation begin",frame_passed)
                        if np.mean(domains_detections["cur_wass_dist"])<(domains_detections["adapt_termination_param"]*np.mean(domains_detections["ini_wass_dist"])) and domains_detections["adaptation"]:
                            domains_detections["adaptation"] = False
                            domains_detections["domain_shift_detected"] = False
                            #domains_detections["validation_frame"] = [[],[]]
                            print("domain adaptation termination",frame_passed)
                        domains_detections["cur_wass_dist"]=domains_detections["cur_wass_dist"][1:]

                domains_detections["storage"] = domains_detections["storage"][domains_detections["storage_length"]:] # detect every storage_temp_length frames
                domains_detections["storage"] = domains_detections["storage"][1:]


            result=[]
            if not domains_detections["adaptation"]:
                result_ori, probs, preds = ema_model(return_loss=False, img=[data['img'][img_id]],
                                                      img_metas=[data['img_metas'][img_id].data[0]])

                domains_detections["storage"].append(np.mean(torch.amax(probs[0], 0).cpu().numpy()))
                result = [preds[0][0].astype(np.int64)]
            else:
                result, probs_, preds_ = anchor_model(return_loss=False, img=[data['img'][img_id]],
                                                      img_metas=[data['img_metas'][img_id].data[0]])  # **data)
                mask = (torch.amax(probs_[0], 0).cpu().numpy() > 0.69).astype(np.int64)
                result, probs, preds = ema_model(return_loss=False, **data)

                result_ori = [(mask * preds[img_id][0] + (1. - mask) * result[0]).astype(np.int64)]
                result = [preds[img_id][0].astype(np.int64)]
                # print(type(probs[0]),probs[0],probs[0].size())
                # print(torch.mean(probs[0]).item())
                conf_mean=np.mean(probs[img_id])
                domains_detections["storage"].append(conf_mean)
                #domains_detections["storage"].append(np.mean(torch.amax(probs[0], 0).cpu().numpy()))



            result_=[result_ori[0].astype(np.int64)]

            weight = 1.

        #if ((new_domain_frame+10)>frame_passed) or round<14: #
        if domains_detections["adaptation"]: #and (len(domains_detections["validation_frame"][0])==domains_detections["num_validation_frame"]):
            #model = deepcopy(ema_model)
            # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #     # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            #     param.data[:] = ema_param[:].data[:]
            if isinstance(result, list):
                if len(data['img'])==14:
                    img_id = 4 #The default size without flip
                else:
                    img_id = 0
                loss = model.forward(return_loss=True, img=data['img'][img_id], img_metas=data['img_metas'][img_id].data[0], gt_semantic_seg=torch.from_numpy(result_[0]).cuda().unsqueeze(0).unsqueeze(0))
                if efficient_test:
                    result = [np2tmp(_) for _ in result]
                results.extend(result)
            else:
                if efficient_test:
                    result = np2tmp(result)
                results.append(result)
            if domains_detections["language_regularization"]:
                torch.mean(loss["decode.loss_seg"]+loss["text_decode.loss_seg"]).backward()
            else:
                torch.mean(loss["decode.loss_seg"]).backward()
            optimizer.step()
            optimizer.zero_grad()

            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=0.999) #teacher model

            #stochastic restoration
            # for nm, m  in model.named_modules():
            # #for nm, m in ema_model.named_modules():
            #     if 'decode_head' in nm or 'backbone' in nm:
            #         for npp, p in m.named_parameters():
            #             if npp in ['weight', 'bias'] and p.requires_grad:
            #                 mask = (torch.rand(p.shape)<0.01).float().cuda()
            #                 with torch.no_grad():
            #                     p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)
        else:
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)


        pred_time += time.time() - pred_begin
    #     batch_size = data['img'][0].size(0)
    #     if i==399:
    #         for _ in range(batch_size):
    #             prog_bar.update()
    #total_predict_time=total_predict_time+pred_time
    print("average pred_time: %.3f seconds;" % (pred_time/(i+1)))
    return results,frame_passed,domains_detections


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
    print("new domain starts,",frame_passed)
    for i, data in enumerate(data_loader):
        model.eval() # student model
        ema_model.eval() # teacher model
        anchor_model.eval() # source model
        frame_passed=frame_passed +1
        ######### Domain resolution selector/Adaptation trigger##################
        if domains_detections["dm_reso_select_processed_frames"] >= domains_detections["hp_k"]:
            if np.mean(domains_detections["dm_reso_select_conf_info"][0]) > np.mean(
                    domains_detections["dm_reso_select_conf_info"][1]):
                domains_detections["imge_id"] = 0
            else:
                domains_detections["imge_id"] = 1
            domains_detections["dm_reso_select_processed_frames"] = -1
            domains_detections["adaptation"] = True
            print("domain resolution selection", domains_detections["imge_id"], np.mean(domains_detections["dm_reso_select_conf_info"][0]) ,np.mean(domains_detections["dm_reso_select_conf_info"][1]) ,round, frame_passed)

        ######### domain shift detection##################
        if len(domains_detections["pred_conf"])>= (2*domains_detections["hp_k"]):
            first_domain_mean=np.mean(list(domains_detections["pred_conf"])[:domains_detections["hp_k"]])
            first_domain_std=np.std(list(domains_detections["pred_conf"])[:domains_detections["hp_k"]])
            second_domain_mean=np.mean(list(domains_detections["pred_conf"])[domains_detections["hp_k"]:])
            second_domain_std=np.std(list(domains_detections["pred_conf"])[domains_detections["hp_k"]:])
            domain_distance=abs(first_domain_mean-second_domain_mean)/np.sqrt(first_domain_std ** 2.0 + second_domain_std ** 2.0)
            print("domain shifted test", domain_distance, first_domain_mean, second_domain_mean, first_domain_std, second_domain_std, frame_passed,round)
            if domain_distance>domains_detections["hp_z_dm_shift"]:
                domains_detections["dm_shift"] = True
                #print("domain shifted",domain_distance, round, frame_passed)

        if domains_detections["dm_shift"]:
            domains_detections["dm_reso_select_processed_frames"] = 0
            domains_detections["dm_reso_select_conf_info"] = [[], []]
            domains_detections["pred_conf"].clear()
            domains_detections["adaptation"] = False
            domains_detections["dm_shift"]=False
        ### we make assumption that frames in the same domain are similar
        elif not domains_detections["dm_shift"] and len(domains_detections["pred_conf"])>= (domains_detections["hp_k"]): ######### Adaptation Termination
            imge_id = 0
            if domains_detections["adaptation"]:
                imge_id = domains_detections["imge_id"]
            source_pred_mean = np.mean(domains_detections["dm_reso_select_conf_info"][imge_id])
            source_pred_std = np.std(domains_detections["dm_reso_select_conf_info"][imge_id])
            teacher_pred_mean = np.mean(list(domains_detections["pred_conf"])[-domains_detections["hp_k"]:])
            teacher_pred_std=np.std(list(domains_detections["pred_conf"])[-domains_detections["hp_k"]:])
            TS_distance=(teacher_pred_mean-source_pred_mean)/np.sqrt(source_pred_std ** 2.0 + teacher_pred_std ** 2.0)
            if TS_distance<domains_detections["hp_z_adapt_ends"] and domains_detections["adaptation"]:
                domains_detections["adaptation"] = False
                domains_detections["pred_conf"].clear()
            if TS_distance > domains_detections["hp_z_adapt_ends"] and not domains_detections["adaptation"]:
                domains_detections["adaptation"] = True
                domains_detections["pred_conf"].clear()
            print("adaptation termination test", domains_detections["adaptation"], TS_distance, source_pred_mean, teacher_pred_mean, source_pred_std,
                      teacher_pred_std, frame_passed, round)
                #print("adaptation termination test",TS_distance, round, frame_passed)

        with torch.no_grad():
            if domains_detections["dm_reso_select_processed_frames"] < 0: #domain resolution has been selectd; start to use teacher model for prediction
                imge_id=0
                if domains_detections["adaptation"]:
                    imge_id = domains_detections["imge_id"]
                result, probs, preds = ema_model(return_loss=False, img=[data['img'][imge_id]],img_metas=[data['img_metas'][imge_id].data[0]])
                domains_detections["pred_conf"].append(np.mean(torch.amax(probs[0], 0).cpu().numpy()))
                #print("teacher pred conf:",np.mean(torch.amax(probs[0], 0).cpu().numpy()))

            ######### domain resolution selector##################
            if domains_detections["dm_reso_select_processed_frames"]>=0 and domains_detections["dm_reso_select_processed_frames"] < domains_detections["hp_k"]:
                result, probs, preds = anchor_model(return_loss=False, **data)
                domains_detections["dm_reso_select_conf_info"][0].append(np.mean(probs[0])) # low reso conf
                domains_detections["dm_reso_select_conf_info"][1].append(np.mean(probs[1])) # high reso conf
                result = [(result[-1]).astype(np.int64)]
                domains_detections["dm_reso_select_processed_frames"]=domains_detections["dm_reso_select_processed_frames"]+1



        if isinstance(result, list):
            if len(data['img']) == 14:
                img_id = 4  # The default size without flip
            else:
                img_id = 0
            if domains_detections["adaptation"]:
                loss = model.forward(return_loss=True, img=data['img'][img_id], img_metas=data['img_metas'][img_id].data[0],
                                     gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0))
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        if domains_detections["adaptation"]:
            torch.mean(loss["decode.loss_seg"]).backward()
            optimizer.step()
            optimizer.zero_grad()
            ema_model = update_ema_variables(ema_model=ema_model, model=model, alpha_teacher=0.999)  # teacher model

    pred_time = time.time() - pred_begin
    print("average pred_time: %.3f seconds;" % (pred_time/(i+1)))
    return results,frame_passed,domains_detections



def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results with GPU."""
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results