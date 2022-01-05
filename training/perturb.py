#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/08/25 11:10 오전
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : perturb.py
# @Software  : PyCharm

import torch
from torch import nn
import tqdm.notebook as tqdm

import numpy as np

import matplotlib.pyplot as plt



def odin_perturb(net, loader, device, epsilon=.1, isTranspose=False):
    net.eval()
    cost_fn = nn.MSELoss()
    X_tildas = None
    grads = None
    X_perturbed = None
    y_total = None

    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        if isTranspose:
            X = X.transpose(2, 1)
        X.requires_grad = True

        prob = net(X)

        cost = cost_fn(prob, y)
        cost.backward()

        grad = X.grad.detach()
        eta = epsilon * torch.sign(-grad)

        X_tilda = X - eta

        if i == 0:
            X_tildas = X_tilda
            grads = grad
            y_total = y
        else:
            X_tildas = torch.cat((X_tildas, X_tilda), dim=0)
            grads = torch.cat((grads, grad), dim=0)
            y_total = torch.cat((y_total, y), dim=0)

    return X_tildas.cpu(), grads.cpu(), X.cpu(), y_total.cpu()

def epsilon_ball_noise(X, epsilon=.1):
    X_pert = X + torch.randn_like(X)
    X_pert = torch.min(X_pert, X+epsilon)
    X_pert = torch.max(X_pert, X-epsilon)
    return X_pert

def plot_grad(g, y_lim, title):
    plt.figure(figsize=(10, 8))
    plt.ylim(y_lim)
    for y_arr, label in zip(g, ['ACCEL', 'CURRENT', 'PEDAL', 'SPEED', 'VOLTAGE']):
        plt.plot(np.arange(0, 100, 1), y_arr, label=label)

    plt.title(title)
    plt.xlabel('Time points')
    plt.ylabel('Mean gradient(abs)')
    plt.legend()
    plt.savefig('results/plots/gradient_%s.png' % title, dpi=300)

    plt.show()
