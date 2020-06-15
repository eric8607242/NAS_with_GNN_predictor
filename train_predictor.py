import random
import argparse

import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.utils import from_networkx

from utils.predictor import Predictor
from utils.config import get_config
from utils.util import get_logger, set_random_seed, save
from utils.graph import calculate_nodes

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
    for row in adj_matrix:
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

    nodes_num = calculate_nodes(CONFIG)

    model = Predictor(nodes_num)
    model = model.cuda()

    optimizer = Adam(params=model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    adj_matrix_table = pd.read_csv(CONFIG.path_to_architecture)
    data = pd.read_csv(CONFIG.path_to_train_data)

    train_data = data.iloc[:250]
    test_data = data.iloc[250:]

    best_loss = 100
    for epoch in range(50):
        for i in range(250):
            optimizer.zero_grad()

            architecture_num = train_data["architecture_num"][i]
            y = train_data["avg"][i]
            adj_matrix = adj_matrix_table.iloc[architecture_num].values
            adj_matrix = adj_matrix.reshape(nodes_num, nodes_num)

            X = get_input_data(adj_matrix)
            edge_index = get_edge_index(adj_matrix)

            X = wrap_data(X)
            y = wrap_data([y])
            edge_index = wrap_data(edge_index, dtype=torch.long)

            outs = model(X, edge_index)
            loss = criterion(outs, y)

            loss.backward()
            optimizer.step()

        test_loss = 0

    
        test_metric = {"architecture_num":[], "predict_avg":[], "avg":[]}
        for i in range(50):
            architecture_num = test_data["architecture_num"][i+250]
            y = test_data["avg"][i+250]
            adj_matrix = adj_matrix_table.iloc[architecture_num].values
            adj_matrix = adj_matrix.reshape(nodes_num, nodes_num)

            X = get_input_data(adj_matrix)
            edge_index = get_edge_index(adj_matrix)

            X = wrap_data(X)
            y = wrap_data([y])
            edge_index = wrap_data(edge_index, dtype=torch.long)

            outs = model(X, edge_index)
            loss = criterion(outs, y)
            test_loss += loss

            test_metric["architecture_num"].append(i+250)
            test_metric["predict_avg"].append(outs.item())
            test_metric["avg"].append(y.item())
        test_loss /= 50
        if best_loss > test_loss.item():
            save(model, "gcn_weight.pth")
            print(test_loss.item())
            df_metric = pd.DataFrame(test_metric)
            df_metric.to_csv("./test.csv", index=False)
            best_loss = test_loss.item()


        

