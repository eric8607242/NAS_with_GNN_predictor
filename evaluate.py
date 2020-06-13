import logging
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn

from utils.config import get_config
from utils.graph import calculate_nodes, get_adj_matrix, get_random_architecture
from utils.util import get_logger, set_random_seed

def generate_architecture(CONFIG, generate_num = 300):
    nodes_num = calculate_nodes(CONFIG)
    
    architecture_metric = []
    for i in range(generate_num):
        adj_matrix = get_adj_matrix(nodes_num, CONFIG)
        adj_matrix = get_random_architecture(adj_matrix, CONFIG)

        architecture = adj_matrix.reshape(-1)
        architecture_metric.append(architecture)

    return architecture_metric

def save_metric(metric, path_to_save_metric):
    df_metric = pd.DataFrame(metric)
    df_metric.to_csv(path_to_save_metric, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="path to the config file", required=True)
    parser.add_argument("--generate-architecture", action="store_true", default=False, help="generate architecture")
    args = parser.parse_args()

    CONFIG = get_config(args.cfg)

    set_random_seed(CONFIG.seed)
    get_logger(CONFIG.log_dir)

    if args.generate_architecture:
        architecture_metric = generate_architecture(CONFIG)
        save_metric(architecture_metric, CONFIG.path_to_architecture)
    
