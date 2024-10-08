import logging

import torch
import torch.optim as optim

from robustbench.data import load_cifar100c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import tent
import norm
import cotta
import time
import ETA
import fastcotta
import rdumb
import OSTTA
import SAR
from sam import SAM

from conf import cfg, load_cfg_fom_args
from torch import nn
import math


logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)
    if cfg.MODEL.ADAPTATION == "rdumb":
        logger.info("test-time adaptation: rdumb")
        model = setup_rdumb(base_model)
    if cfg.MODEL.ADAPTATION == "OSTTA":
        logger.info("test-time adaptation: OSTTA")
        model = setup_ostta(base_model)
    if cfg.MODEL.ADAPTATION == "fastcotta":
        logger.info("test-time adaptation: FastCoTTA")
        model = setup_fastcotta(base_model)
    if cfg.MODEL.ADAPTATION == "ETA":
        logger.info("test-time adaptation: ETA")
        model = setup_ETA(base_model)
    if cfg.MODEL.ADAPTATION == "SAR":
        logger.info("test-time adaptation: SAR")
        model = setup_SAR(base_model)
    # evaluate on each severity and type of corruption in turn
    prev_ct = "x0"
    pred_time=0
    average_acc=0
    dataset_count=0
    model.reset()
    logger.info("resetting model")
    for i in range(10):
        for severity in cfg.CORRUPTION.SEVERITY:
            for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
                # continual adaptation for all corruption
                # if i_c == 0:
                #     try:
                #         model.reset()
                #         logger.info("resetting model")
                #     except:
                #         logger.warning("not resetting model")
                # else:
                #     logger.warning("not resetting model")
                x_test, y_test = load_cifar100c(cfg.CORRUPTION.NUM_EX,
                                               severity, cfg.DATA_DIR, False,
                                               [corruption_type])
                pred_begin = time.time()
                x_test, y_test = x_test.cuda(), y_test.cuda()
                acc = my_accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE, cfg.MODEL.ADAPTATION)
                err = 1. - acc
                pred_begin = time.time() - pred_begin
                pred_time = pred_time + pred_begin
                average_acc = average_acc + acc
                dataset_count = dataset_count + 1
                logger.info(f"acc % [{corruption_type}{severity}]: {acc:.2%}")
    print("method:%s; average accuracy: %.3f;total pred time:%.3f seconds; " % (
    cfg.MODEL.ADAPTATION, average_acc / dataset_count, pred_time))


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC, 
                           mt_alpha=cfg.OPTIM.MT, 
                           rst_m=cfg.OPTIM.RST, 
                           ap=cfg.OPTIM.AP)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

def setup_fastcotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = fastcotta.configure_model(model)
    params, param_names = fastcotta.collect_params(model)
    optimizer = setup_optimizer(params)
    fastcotta_model = fastcotta.FastCoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           mt_alpha=cfg.OPTIM.MT,
                           rst_m=cfg.OPTIM.RST,
                           ap=cfg.OPTIM.AP)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return fastcotta_model

def setup_rdumb(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = rdumb.configure_model(model)
    params, param_names = rdumb.collect_params(model)
    optimizer = setup_optimizer(params)
    rdumb_model = rdumb.RDumb(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           mt_alpha=cfg.OPTIM.MT,
                           rst_m=cfg.OPTIM.RST,
                           ap=cfg.OPTIM.AP)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return rdumb_model

def setup_ostta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """

    model = OSTTA.configure_model(model)
    params, param_names = OSTTA.collect_params(model)
    optimizer = setup_optimizer(params)
    OSTTA_model = OSTTA.OSTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)

    return OSTTA_model

def setup_ETA(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = ETA.configure_model(model)
    params, param_names = ETA.collect_params(model)
    optimizer = setup_optimizer(params)
    ETA_model = ETA.EATA(model, optimizer, steps=cfg.OPTIM.STEPS,episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return ETA_model

def setup_SAR(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = SAR.configure_model(model)
    params, param_names = SAR.collect_params(model)
    if cfg.OPTIM.METHOD == 'Adam':
        base_optimizer = optim.Adam
        optimizer = SAM(params, base_optimizer,lr=cfg.OPTIM.LR,betas=(cfg.OPTIM.BETA, 0.999),weight_decay=cfg.OPTIM.WD)
    if cfg.OPTIM.METHOD == 'SGD':
        base_optimizer = optim.SGD
        optimizer = SAM(params, base_optimizer, lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM,dampening=cfg.OPTIM.DAMPENING,
                        weight_decay=cfg.OPTIM.WD,nesterov=cfg.OPTIM.NESTEROV)

    SAR_model = SAR.SAR(model, optimizer)

    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return SAR_model

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


def my_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   adaptation_method=None,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    passed_batches=0
    with torch.no_grad():
        for counter in range(n_batches):
            passed_batches = passed_batches+1
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            if adaptation_method == "fastcotta" or adaptation_method == "rdumb":
                output = model(x_curr, passed_batches)
            else:
                output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()
    return acc.item() / x.shape[0]



if __name__ == '__main__':
    evaluate('"CIFAR-100-C evaluation.')
