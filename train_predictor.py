import random
import argparse

import torch
from torch.optim import Adam
from torch_geometric.utils import from_networkx

from utils.predictor import Predictor
from utils.config import get_config
from utils.util import get_logger, set_random_seed

def get_edge_index(adj_matrix):
    edge_index = [[], []]
    for i, row in enumerate(adj_matrix):
        for j, v in enumerate(row):
            if v == 1:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return edge_index

def get_input_data(adj_matrix):
    degree_value = []
    for row in enumerate(adj_matrix):
        degree_value.append([sum(row)-1, 1, 1])
    return degree_value

def wrap_data(data, dtype=None, cuda=True):
    data = torch.Tensor(data) if dtype is None else torch.tensor(data, dtype=dtype)
    data = data.cuda() if cuda else data
    return data

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    CONFIG = get_config(args.cfg)

    if CONFIG.cuda:
        device = torch.device("cuda" if (torch.cuda.is_available() and CONFIG.ngpu > 0) else "cpu")
    else:
        device = torch.device("cpu")

    set_random_seed(CONFIG.seed)

    model = Predictor()
    model = model.cuda()

    optimizer = Adam(params=model.parameters(), lr=0.0001)

    data = pd.read_csv(CONFIG.path_to_architecture)
    adj_matrix_table = pd.read_csv(CONFIG.path_to_train_data)

    train_data = data.iloc[:250]
    test_data = data.iloc[250:]

    for epoch in range(1):
        architecture_num = train_data.iloc[0]["architecture_num"]
        adj_matrix = adj_matrix_table.iloc[architecture_num].values
        adj_matrix = adj_matrix.reshape(nodes_num, nodes_num)

        X = get_input_data(adj_matrix)
        edge_index = get_edge_index(adj_matrix)

        X = wrap_data(X)
        edge_index = wrap_data(edge_index)

        outs = model(X, edge_index)

        

