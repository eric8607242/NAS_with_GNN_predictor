import logging 

import torch
import torch.nn as nn

def get_lr_scheduler(optimizer, step_per_epoch, CONFIG):
    if CONFIG.lr_scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_per_epoch*CONFIG.epochs)
    elif CONFIG.lr_scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG.step_size, gamma=CONFIG.decay_ratio, last_epoch=-1)

    return lr_scheduler
    


def get_optimizer(model, CONFIG, log_info=""):
    if CONFIG.optimizer == "sgd":
        logging.info("{} optimizer: SGD".format(log_info))
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=CONFIG.lr,
                                    momentum=CONFIG.momentum,
                                    weight_decay=CONFIG.weight_decay)

    elif CONFIG.optimizer == "rmsprop":
        logging.info("{} optimizer: RMSprop".format(log_info))
        optimizer = torch.optim.RMSprop(model.parameters(),
                            lr=CONFIG.lr,
                            alpha=CONFIG.alpha,
                            momentum=CONFIG.momentum,
                            weight_decay=CONFIG.weight_decay)
    elif CONFIG.optimizer == "adam":
        logging.info("{} optimizer: Adam".format(log_info))
        optimizer = torch.optim.Adam(model.parameters(),
                            weight_decay=CONFIG.weight_decay,
                            lr=CONFIG.lr,
                            betas=(CONFIG.beta, 0.999))



    return optimizer


def cal_hc_loss(generate_hc, target_hc, alpha, loss_penalty):
    if generate_hc > target_hc + 0.1:
        return (generate_hc-target_hc)**2 * alpha * loss_penalty
    elif generate_hc < target_hc - 0.1:
        return (target_hc-generate_hc)**2 * alpha
    else:
        return torch.tensor(0)
