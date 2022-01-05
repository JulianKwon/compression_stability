#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/17 2:20 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : models.py
# @Software  : PyCharm

import torch.nn as nn


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.kaiming_uniform_(self.conv1.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight.data, mode='fan_in', nonlinearity='relu')

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
        )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Generator(nn.Module):
    def __init__(self, z_dim, gen_size=256, channels=3):
        super(Generator, self).__init__()
        self.gen_size = gen_size
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * gen_size)
        self.final = nn.Conv2d(gen_size, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform(self.final.weight.data, nn.init.calculate_gain('tanh'))

        self.model = nn.Sequential(
            ResBlockGenerator(gen_size, gen_size, stride=2),
            ResBlockGenerator(gen_size, gen_size, stride=2),
            ResBlockGenerator(gen_size, gen_size, stride=2),
            nn.BatchNorm2d(gen_size),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, self.gen_size, 4, 4))

