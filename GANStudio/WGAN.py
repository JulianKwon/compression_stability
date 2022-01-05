#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/15 7:50 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : WGAN.py
# @Software  : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from scipy.stats import truncnorm
import numpy as np

from data import cifar10


def init_weights(modules, initialize):
    for module in modules():
        if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.ConvTranspose2d)
                or isinstance(module, nn.Linear)):
            if initialize == 'ortho':
                init.orthogonal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize == 'N02':
                init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize in ['glorot', 'xavier']:
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            else:
                print('Init style not recognized...')
        elif isinstance(module, nn.Embedding):
            if initialize == 'ortho':
                init.orthogonal_(module.weight)
            elif initialize == 'N02':
                init.normal_(module.weight, 0, 0.02)
            elif initialize in ['glorot', 'xavier']:
                init.xavier_uniform_(module.weight)
            else:
                print('Init style not recognized...')
        else:
            pass


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_fn):
        super(GenBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.conv2d0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, label):
        x0 = x

        x = self.bn1(x)
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv2d1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest')
        x0 = self.conv2d0(x0)

        out = x + x0
        return out

class Generator(nn.Module):
    def __init__(self, z_dim, img_size, g_conv_dim, activation_fn, num_classes, initialize):
        super(Generator, self).__init__()
        g_in_dims_collection = {"32": [g_conv_dim*4, g_conv_dim*4, g_conv_dim*4],
                                "64": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "128": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "256": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "512": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim]}

        g_out_dims_collection = {"32": [g_conv_dim*4, g_conv_dim*4, g_conv_dim*4],
                                 "64": [g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "128": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "256": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "512": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim, g_conv_dim]}
        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.z_dim = z_dim
        self.num_classes = num_classes
        conditional_bn = False

        self.in_dims =  g_in_dims_collection[str(img_size)]
        self.out_dims = g_out_dims_collection[str(img_size)]
        self.bottom = bottom_collection[str(img_size)]

        self.linear0 = nn.Linear(in_features=self.z_dim, out_features=self.in_dims[0]*self.bottom*self.bottom)

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[GenBlock(in_channels=self.in_dims[index],
                                          out_channels=self.out_dims[index],
                                          activation_fn=activation_fn)]]

            # if index+1 == attention_after_nth_gen_block and attention is True:
            #     self.blocks += [[Self_Attn(self.out_dims[index], g_spectral_norm)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.bn4 = nn.BatchNorm2d(self.out_dims[-1])

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.conv2d5 = nn.Conv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)

        self.tanh = nn.Tanh()

        # Weight init
        if initialize is not False:
            init_weights(self.modules, initialize)

    def forward(self, z, label, evaluation=False):
        act = self.linear0(z)
        act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                act = block(act, label)
        act = self.bn4(act)
        act = self.activation(act)
        act = self.conv2d5(act)
        out = self.tanh(act)
        return out


def sample_latents(dist, batch_size, dim, truncated_factor=-1.0, num_classes=None, perturb=None, device=torch.device("cpu"), sampler="default"):
    if num_classes:
        if sampler == "default":
            y_fake = torch.randint(low=0, high=num_classes, size=(batch_size,), dtype=torch.long, device=device)
        elif sampler == "class_order_some":
            assert batch_size % 8 == 0, "The size of the batches should be a multiple of 8."
            num_classes_plot = batch_size//8
            indices = np.random.permutation(num_classes)[:num_classes_plot]
        elif sampler == "class_order_all":
            batch_size = num_classes*8
            indices = [c for c in range(num_classes)]
        elif isinstance(sampler, int):
            y_fake = torch.tensor([sampler]*batch_size, dtype=torch.long).to(device)
        else:
            raise NotImplementedError

        if sampler in ["class_order_some", "class_order_all"]:
            y_fake = []
            for idx in indices:
                y_fake += [idx]*8
            y_fake = torch.tensor(y_fake, dtype=torch.long).to(device)
    else:
        y_fake = None

    if isinstance(perturb, float) and perturb > 0.0:
        if dist == "gaussian":
            latents = sample_normal(batch_size, dim, truncated_factor, device)
            latents_eps = latents + perturb*sample_normal(batch_size, dim, -1.0, device)
        elif dist == "uniform":
            latents = torch.FloatTensor(batch_size, dim).uniform_(-1.0, 1.0).to(device)
            latents_eps = latents + perturb*torch.FloatTensor(batch_size, dim).uniform_(-1.0, 1.0).to(device)
        else:
            raise NotImplementedError
        return latents, y_fake, latents_eps
    else:
        if dist == "gaussian":
            latents = sample_normal(batch_size, dim, truncated_factor, device)
        elif dist == "uniform":
            latents = torch.FloatTensor(batch_size, dim).uniform_(-1.0, 1.0).to(device)
        else:
            raise NotImplementedError
        return latents, y_fake



def sample_normal(batch_size, dim, truncated_factor, device):
    if truncated_factor == -1.0:
        latents = torch.randn(batch_size, dim, device=device)
    elif truncated_factor > 0:
        latents = torch.FloatTensor(truncated_normal([batch_size, dim], truncated_factor)).to(device)
    else:
        raise ValueError("truncated_factor must be positive.")
    return latents

def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

def loss_wgan_gen(gen_out_fake):
    return -torch.mean(gen_out_fake)

def loss_wgan_dis(dis_out_real, dis_out_fake):
    return torch.mean(dis_out_fake - dis_out_real)


if __name__ == '__main__':
    # def train():
    CUDA_NUM = 0
    base_path = ''
    model_args = {
        "z_dim": 128,
        "img_size": 32,
        "g_conv_dim": 64,
        "activation_fn": "ReLU",
        "num_classes": 10,
        "g_init": "ortho",
    }
    optim_args = {
        "lr": 0.0002,
        "weight_decay": 0.0,
        "betas": [0.5, 0.999],
        "eps": 1e-6
    }

    train_args = {
        "prior": "gaussian",
        'batch_size': 128,
    }

    train_set, test_set, _ = cifar10(normalization=False, data_root='/Users/junhyungkwon/Projects/dataset/')

    gen_model = Generator(**model_args)
    dis_model = get_classifier(base_path, 0.)

    gen_model.train()

    optimizer = torch.optim.Adam(gen_model.parameters(), **optim_args)
    criterion = loss_wgan_gen

    # for epoch in range(50):

    zs, fake_labels = sample_latents(train_args['prior'], train_args['batch_size'], model_args['z_dim'],
                                     -1.0, model_args['num_classes'], None, CUDA_NUM)

    for real_images, real_labels in train_set:
        break

    fake_images = gen_model(zs, fake_labels)
    dis_out_real = dis_model(real_images, real_labels)
    dis_out_fake = dis_model(fake_images, fake_labels)
