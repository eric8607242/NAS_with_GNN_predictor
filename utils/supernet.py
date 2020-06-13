import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.network_utils import *


class Layer(nn.Module):
    def __init__(self, input_channel, output_channel, stride, op_setting, CONFIG, affine=True):
        super(Layer, self).__init__()

        op_list = []
        for name, op_flag in zip(CONFIG.ops_name, op_setting):
            if op_flag == 1:
                op_list.append(OPS[name](input_channel, output_channel, stride, affine))

        self.ops = nn.ModuleList(op_list)

    def forward(self, x):
        outs = None
        for op in self.ops:
            out = op(x)
            outs = out if outs is None else outs + out

        return outs



class Supernet(nn.Module):
    def __init__(self, adj_matrix, CONFIG):
        super(Supernet, self).__init__()
        
        self.CONFIG = CONFIG
        self.classes = CONFIG.classes
        self.dataset = CONFIG.dataset

        if self.dataset[:5] == "cifar":
            self.first = ConvBNRelu(input_channel=3, output_channel=16, kernel=3, stride=1,
                                    pad=3//2, activation="relu")

        ops_num = len(self.CONFIG.ops_name)
        ops_list = []
        input_channel = 16
        for l, l_cfg in enumerate(self.CONFIG.l_cfgs):
            output_channel, stride = l_cfg
            op_setting = adj_matrix[l*(ops_num+1), l*(ops_num+1)+1:(l+1)*(ops_num+1)]
            ops_list.append(Layer(input_channel, output_channel, stride, op_setting, CONFIG))

            input_channel = output_channel

        self.stages = nn.ModuleList(ops_list)
        self.last_stage = conv_1x1_bn(input_channel, 1280)
        self.classifier = nn.Linear(1280, self.classes)


    def forward(self, x):
        y = self.first(x)
        for l in self.stages:
            y = l(y)

        y = self.last_stage(y)
        y = y.mean(3).mean(2)
        y = self.classifier(y)

        return y

            





            



