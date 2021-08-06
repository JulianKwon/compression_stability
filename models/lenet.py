from collections import OrderedDict

import torch.nn as nn

from .l0_layers import *


class L0LeNet(nn.Module):
    def __init__(self, num_classes, device=0, input_size=(1, 28, 28), conv_dims=[32, 64], fc_dims=[256, 256],
                 kernel_sizes=[5, 5], layer_sparsity=(0., 0., 0., 0.)):
        super(L0LeNet, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.kernel_sizes = kernel_sizes
        self.layer_sparsity = layer_sparsity

        if torch.cuda.is_available():
            self.device = 'cuda:%d' % device
        else:
            self.device = 'cpu'

        # model structure
        self.model_construct()

    def lamb(self):
        l = []
        for layer in self.layers:
            l.append(layer.get_lamb())
        return l

    def model_construct(self, sparsity=None):
        if sparsity is not None:
            self.layer_sparsity = sparsity

        convs = OrderedDict([
            ('conv1',
             L0Conv2d(self.input_size[0], self.conv_dims[0], self.kernel_sizes[0], sparse_ratio=self.layer_sparsity[0],
                      device=self.device)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(2)),
            ('conv2',
             L0Conv2d(self.conv_dims[0], self.conv_dims[1], self.kernel_sizes[1], sparse_ratio=self.layer_sparsity[1],
                      device=self.device)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(2))
        ])

        self.convs = nn.Sequential(convs)

        h_out = lambda: int(self.convs.conv2.h_out((self.convs.conv1.h_out(self.input_size[1]) / 2)) / 2)
        w_out = lambda: int(self.convs.conv2.w_out((self.convs.conv1.w_out(self.input_size[1]) / 2)) / 2)
        self.output_len = self.conv_dims[-1] * h_out() * w_out()

        if torch.cuda.is_available():
            self.convs = self.convs.to(self.device)

        fcs = OrderedDict([
            ('fc1', L0Dense(self.output_len, self.fc_dims[0], sparse_ratio=self.layer_sparsity[2], device=self.device)),
            ('relu3', nn.ReLU()),
            ('fc2', L0Dense(self.fc_dims[0], self.fc_dims[1], sparse_ratio=self.layer_sparsity[3], device=self.device)),
            ('relu4', nn.ReLU()),
            ('fc3', L0Dense(self.fc_dims[1], self.num_classes, sparse_ratio=self.layer_sparsity[3], device=self.device)),
        ])
        self.fcs = nn.Sequential(fcs)

        if torch.cuda.is_available():
            self.fcs = self.fcs.to(self.device)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense) or isinstance(m, L0Conv2d):
                self.layers.append(m)

    def forward(self, input):
        out = self.convs(input)
        out = out.view(out.size(0), -1)
        out = self.fcs(out)
        return out
