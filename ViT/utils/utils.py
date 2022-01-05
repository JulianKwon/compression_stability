#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/02 7:37 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : utils.py
# @Software  : PyCharm


import os
import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from torch.distributions import kl_divergence
from tqdm import tqdm_notebook as tqdm
import torch.distributions.dirichlet



def get_capacity(net):
    total_zeros = 0
    non_zeros = 0
    capacity = 0

    for name, parameters in net.named_parameters():
        if 'weight' in name:
            if 'conv' in name:
                total_zeros += parameters[parameters == 0].shape[0]
                non_zeros += parameters[parameters != 0].shape[0]
                capacity += non_zeros
            #                 print(name, non_zeros)
            else:
                #             if isinstance(module, torch.nn.Linear):
                #                 print(name, np.prod(parameters.shape))
                capacity += np.prod(parameters.shape)
    return capacity, total_zeros


def cal_kl(state, uniform):
    total_kl = 0
    ran = state['softmax_output'].shape[0]
    for i in range(ran):
        total_kl += kl_divergence(state['softmax_output'][i], uniform)

    return total_kl / ran


def get_weights_copy(model, device, weights_path='tmp/weights_temp.pt'):
    torch.save(model.cpu().state_dict(), weights_path)
    model.to(device)
    return torch.load(weights_path)


def accuracy(y_data, p_data, s_data):
    """Computes the precision@k for the specified values of k"""

    f1 = f1_score(y_data, p_data, average='micro')
    acc = accuracy_score(y_data, p_data)
    auc = roc_auc_score(y_data, s_data, multi_class='ovo')
    pre = precision_score(y_data, p_data, average='micro')
    rec = recall_score(y_data, p_data, average='micro')

    return {
        'f1': f1,
        'accuracy': acc,
        'auc_score': auc,
        'precision': pre,
        'recall': rec
    }


def reshape_resulting_array(arr):
    return arr.reshape((-1, arr.shape[-1]))


def get_flat_fts(in_size, fts, device):
    dummy_input = torch.ones(1, *in_size)
    if torch.cuda.is_available():
        dummy_input = dummy_input.to(device)
    f = fts(torch.autograd.Variable(dummy_input))
    print('conv_out_size: {}'.format(f.size()))
    return int(np.prod(f.size()[1:]))


def save_checkpoint(state, name, is_best=False, filename='checkpoint'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % name
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_d = directory + filename
    torch.save(state, '%s.pth.tar' % file_d)
    if is_best:
        shutil.copyfile('%s.pth.tar' % file_d, 'runs/%s/%s_best.pth.tar' % (name, filename))


def save_data(state, directory, filename='inference_result.pkl'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    with open(filename, 'wb') as f:
        pickle.dump(state, f)


def load_data(file_path):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)
    return file


def plot_grad(g):
    plt.figure(figsize=(10, 8))
    for y_arr, label in zip(g, ['ACCEL', 'CURRENT', 'PEDAL', 'SPEED', 'VOLTAGE']):
        plt.plot(np.arange(0, 100, 1), y_arr, label=label)

    plt.legend()
    plt.show()


def get_grad(net, loader, criterion, device):
    net.eval()
    grads = None

    for i, (X, y) in tqdm(enumerate(loader)):
        X, y = X.to(device), y.to(device)
        X = X.transpose(2, 1)
        X.requires_grad = True
        prob = net(X)
        cost = criterion(prob, y)
        cost.backward()

        grad = X.grad.detach()

        if i == 0:
            grads = grad
        else:
            grads = torch.cat((grads, grad), dim=0)

    return grads.cpu()


def odin_perturb(net, loader, device, epsilon=.1):
    net.eval()
    cost_fn = nn.MSELoss()
    X_tildas = None
    grads = None

    for i, (X, y) in tqdm(enumerate(loader)):
        X, y = X.to(device), y.to(device)
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
        else:
            X_tildas = torch.cat((X_tildas, X_tilda), dim=0)
            grads = torch.cat((grads, grad), dim=0)

    return X_tildas.cpu(), grads.cpu(), X.cpu(), y.cpu()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
