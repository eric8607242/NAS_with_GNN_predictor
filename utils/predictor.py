import numpy as np

import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, input_channels, output_channels):
        super(GCNConv, self).__init__(aggr="add")

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index=edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1)*x_j

    def update(self, aggr_out):
        return aggr_out

class Encoder_Block(nn.Module):
    def __init__(self, dim=4):
        super(Encoder_Block, self).__init__()
        self.gcn = GCNConv(dim, dim)
        self.gru = nn.GRU(input_size=dim, hidden_size=dim)

    def forward(self, x, edge_index):
        h_n = self.gcn(x, edge_index)

        h_n = h_n.view(1, *h_n.shape)
        x = x.view(1, *x.shape)
        h_v = self.gru(h_n, x)

        return h_v[0][0]

class Encoder(nn.Module):
    def __init__(self, input_dim=3, layer_nums=5, embedding_dim=4):
        super(Encoder, self).__init__()
        self.first = nn.Linear(input_dim, embedding_dim)
        self.relu = nn.LeakyReLU()
        self.layers = nn.ModuleList()

        for l in range(layer_nums):
            self.layers.append(Encoder_Block(embedding_dim))

    def forward(self, x, edge_index):
        outs = []
        x = self.first(x)
        x = self.relu(x)
        for l in self.layers:
            x = l(x, edge_index)
            outs.append(x)
        outs = torch.stack(outs)
        max_out = outs.max(dim=0).values
        return max_out

class Predictor(nn.Module):
    def __init__(self, nodes_num, input_dim=3, embedding_dim=128, hidden_dim=54, layer_nums=5):
        super(Predictor, self).__init__()
        self.e = Encoder(input_dim, layer_nums, embedding_dim=embedding_dim)

        self.predictor = nn.Sequential(nn.Linear(nodes_num*embedding_dim, hidden_dim),
                                       nn.LeakyReLU(),
                                       nn.Linear(hidden_dim, 1),
                                       nn.Sigmoid())
        self.nodes_num = nodes_num
        self.embedding_dim = embedding_dim

    def forward(self, x, edge_index):
        x = self.e(x, edge_index)
        x = x.reshape(-1, self.nodes_num*self.embedding_dim)
        x = self.predictor(x)
        return x

if __name__ == "__main__":
    model = GCNConv(4, 4)

    x = torch.tensor([[2, 1, 1], [2, 1, 1], [2, 1, 1], [2, 1, 1]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2],
                               [1, 2, 3]], dtype=torch.long)

    print(model(x, edge_index))
