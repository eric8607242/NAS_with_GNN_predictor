import time
import logging
import argparse

import torch
import torch.nn as nn

from utils.supernet import Supernet
from utils.config import get_config
from utils.util import get_writer, get_logger, set_random_seed, cross_encropy_with_label_smoothing, cal_model_efficient
from utils.dataflow import get_transforms, get_dataset, get_dataloader
from utils.optim import get_optimizer, get_lr_scheduler
from utils.trainer import Trainer
from utils.graph import calculate_nodes, get_adj_matrix, get_random_architecture


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="path to the config file", required=True)
    args = parser.parse_args()

    CONFIG = get_config(args.cfg)

    if CONFIG.cuda:
        device = torch.device("cuda" if (torch.cuda.is_available() and CONFIG.ngpu > 0) else "cpu")
    else:
        device = torch.device("cpu")

    set_random_seed(CONFIG.seed)

    get_logger(CONFIG.log_dir)

    nodes_num = calculate_nodes(CONFIG)
    adj_matrix = get_adj_matrix(nodes_num, CONFIG)
    adj_matrix = get_random_architecture(adj_matrix, CONFIG)
    print(adj_matrix)

    train_transform, val_transform, test_transform = get_transforms(CONFIG)
    train_dataset, val_dataset, test_dataset = get_dataset(train_transform, val_transform, test_transform, CONFIG)
    train_loader, val_loader, test_loader = get_dataloader(train_dataset, val_dataset, test_dataset, CONFIG)

    model = Supernet(adj_matrix, CONFIG)
    model = model.to(device)
    if (device.type == "cuda" and CONFIG.ngpu >= 1):
        model = nn.DataParallel(model, list(range(CONFIG.ngpu)))

    criterion = cross_encropy_with_label_smoothing
    cal_model_efficient(model, CONFIG)
    # ============================

    optimizer = get_optimizer(model, CONFIG.optim_state)
    scheduler = get_lr_scheduler(optimizer, len(train_loader), CONFIG)

    start_time = time.time()
    trainer = Trainer(criterion, optimizer, scheduler, None, device, CONFIG)
    trainer.train_loop(train_loader, test_loader, model)
    logging.info("Total training time : {:.2f}".format(time.time() - start_time))
    

