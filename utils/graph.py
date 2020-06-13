import random

import numpy as np

import torch
import torch.nn as nn

def calculate_nodes(CONFIG):
    layers_num = len(CONFIG.l_cfgs)
    ops_num = len(CONFIG.ops_name)

    return layers_num * ops_num + layers_num + 1

def get_adj_matrix(nodes_num, CONFIG):
    adj_matrix = np.identity(nodes_num)

    return adj_matrix

def get_random_architecture(adj_matrix, CONFIG):
    ops_num = len(CONFIG.ops_name)
    for l in range(len(CONFIG.l_cfgs)):
        layer_connect = np.random.uniform(0, 1, ops_num)
        layer_connect = (layer_connect > 0.5).astype(int)

        if sum(layer_connect) == 0:
            # Add at least one edge
            layer_connect[random.randint(0, ops_num-1)] = 1

        adj_matrix[l*(ops_num+1), l*(ops_num+1)+1:(l+1)*(ops_num+1)] = layer_connect
        adj_matrix[l*(ops_num+1)+1:(l+1)*(ops_num+1), (l+1)*(ops_num+1)] = layer_connect
    return adj_matrix


