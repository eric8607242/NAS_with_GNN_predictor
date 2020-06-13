import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
        "skip" : lambda input_channel, output_channel, stride, affine: Identity() if (stride==1 and input_channel == output_channel) else FactorizedReduce(input_channel, output_channel, affine=affine),
        "sep_conv_3x3" : lambda input_channel, output_channel, stride, affine: SepConv(input_channel, output_channel, 3, stride, 1, affine=affine),
        "sep_conv_5x5" : lambda input_channel, output_channel, stride, affine: SepConv(input_channel, output_channel, 5, stride, 2, affine=affine),
        "sep_conv_7x7" : lambda input_channel, output_channel, stride, affine: SepConv(input_channel, output_channel, 7, stride, 3, affine=affine),
        "dil_conv_3x3" : lambda input_channel, output_channel, stride, affine: DilConv(input_channel, output_channel, 3, stride, 2, 2, affine=affine),
        "dil_conv_5x5" : lambda input_channel, output_channel, stride, affine: DilConv(input_channel, output_channel, 5, stride, 4, 2, affine=affine),
        }


class DilConv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=input_channel, bias=False),
                    nn.Conv2d(input_channel, output_channel, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm2d(output_channel, affine=affine)
                )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=input_channel, bias=False),
                    nn.Conv2d(input_channel, input_channel, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm2d(input_channel, affine=affine),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=1, padding=padding, groups=input_channel, bias=False),
                    nn.Conv2d(input_channel, output_channel, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm2d(output_channel, affine=affine),
                )

    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FactorizedReduce(nn.Module):
    def __init__(self, input_channel, output_channel, affine=True):
        super(FactorizedReduce, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(input_channel, output_channel//2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(input_channel, output_channel//2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(output_channel, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out

class ConvBNRelu(nn.Sequential):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel,
                 stride,
                 pad,
                 activation="relu",
                 bn=True,
                 group=1,
                 *args,
                 **kwargs):

        super(ConvBNRelu, self).__init__()

        assert activation in ["hswish", "relu", None]
        assert stride in [1, 2, 4]

        self.add_module("conv", nn.Conv2d(input_channel, output_channel, kernel, stride, pad, groups=group, bias=False))
        if bn:
            self.add_module("bn", nn.BatchNorm2d(output_channel))

        if activation == "relu":
            self.add_module("relu", nn.ReLU6(inplace=True))

def conv_1x1_bn(input_channel, output_channel):
    return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channel),
                nn.ReLU6(inplace=True)
            )
