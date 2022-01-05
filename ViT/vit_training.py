#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/07 3:50 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : vit_training.py
# @Software  : PyCharm

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dataloader
from model.pytorch_pretrained_vit import ViT
from regularizer import prune_vit
from utils import AverageMeter, get_weights_copy, reshape_resulting_array, accuracy

BASE_PATH = '/workspace/paper_works/work_results/'

def train_iter(net, loader, criterion, optimizer, epoch, device, reg_ratio=0., print_freq=200):
    losses = AverageMeter()
    batch_time = AverageMeter()
    net.train()
    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        pred = net(X)
        loss = criterion(pred, y)

        losses.update(loss.data, X.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if reg_ratio > 0.:
            idxs, lams = prune_vit(net, reg_ratio)

        if i % print_freq == 0 or len(loader) - 1 == i:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader), loss=losses))

    if reg_ratio > 0.:
        idxs, lams = prune_vit(net, reg_ratio)

    state_dicts = get_weights_copy(net, device, weights_path='tmp/weights_temp.pt')

    state = {
        'epoch': epoch + 1,
        'state_dict': state_dicts,
        'loss': losses.avg.detach().cpu().numpy().tolist(),
    }

    if reg_ratio > 0.:
        state['reg_idxs'] = idxs
        state['reg_lams'] = lams

    return state


def valid_iter(net, loader, criterion, epoch, device, return_result=True, print_freq=100):
    losses = AverageMeter()
    net.eval()

    total_softmax = []
    total_labels = []
    total_preds = []

    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        pred = net(X)
        loss = criterion(pred, y)

        losses.update(loss.data, X.size(0))

        if return_result:
            pred = F.softmax(pred)
            _, p_data = pred.data.max(dim=1)
            p_data = p_data.cpu().detach().numpy()
            s_data, y_data = pred.data.cpu().detach().numpy(), y.data.cpu().detach().numpy()

            if i == 0:
                total_preds = p_data
                total_softmax = s_data
                total_labels = y_data
            else:
                total_preds = np.concatenate((total_preds, p_data))
                total_softmax = np.concatenate((total_softmax, s_data))
                total_labels = np.concatenate((total_labels, y_data))

        if i % print_freq == 0 or len(loader) - 1 == i:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader), loss=losses))

    state = {
        'loss': losses.avg.detach().cpu().numpy().tolist(),
    }

    if return_result:

        if type(total_softmax) is not np.ndarray:
            total_softmax = np.array(total_softmax)
            total_labels = np.array(total_labels)
            total_preds = np.array(total_preds)

        total_softmax = reshape_resulting_array(total_softmax)
        total_preds = total_preds.reshape(-1).squeeze()
        total_labels = total_labels.reshape(-1).squeeze()

        print(total_softmax.shape, total_preds.shape, total_labels.shape)

        state['softmax_output'] = total_softmax
        state['labels'] = total_labels
        state['results'] = accuracy(total_labels, total_preds, total_softmax)
        state['preds'] = total_preds
    return state


def main(device, net_name, dataset_args, dataset_name, early_stopping_step,
         lr, model_args, model_name, EPOCH=100, reg_ratio=0.):
    if not os.path.exists('{}/{}_{}/'.format(BASE_PATH, dataset_name, net_name)):
        os.mkdir('{}/{}_{}/'.format(BASE_PATH, dataset_name, net_name))

    file_name = 'best_state.ptl'
    if reg_ratio == 0:
        net = ViT(model_name, True, **model_args)
    else:
        net = ViT(model_name, False, **model_args)
        state = torch.load('{}/{}_{}/{}'.format(BASE_PATH, dataset_name, net_name, file_name))
        net.load_state_dict(state['state_dict'])
        #         net = ViT(model_name, True, **model_args)
        file_name = 'best_state_ratio%d.ptl' % int(reg_ratio * 100)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    trainset, testset, num_classes = getattr(dataloader, dataset_name)(**dataset_args)

    best_state = None
    best_loss = 9999999
    early_stopping_idx = 0
    for epoch in range(EPOCH):
        early_stopping_idx += 1
        train_state = train_iter(net, trainset, criterion, optimizer, epoch, device, reg_ratio)
        valid_state = valid_iter(net, testset, criterion, epoch, device, return_result=False)

        if best_loss > valid_state['loss']:
            best_loss = valid_state['loss']
            best_state = train_state
            torch.save(best_state, '{}/{}_{}/{}'.format(BASE_PATH, dataset_name, net_name, file_name))

            early_stopping_idx = 0

        if early_stopping_idx > early_stopping_step:
            print('early stopping.')
            break

    del net


def inference(device, net_name, dataset_args, dataset_name, early_stopping_step,
              lr, model_args, model_name, EPOCH=100, reg_ratio=0.):
    criterion = nn.CrossEntropyLoss()
    net = ViT(model_name, False, **model_args)

    if reg_ratio == 0:
        file_name = 'best_state.ptl'
    else:
        file_name = 'best_state_ratio%d.ptl' % int(reg_ratio * 100)

    state = torch.load('{}/{}_{}/{}'.format(BASE_PATH, dataset_name, net_name, file_name))
    net.load_state_dict(state['state_dict'])
    net.to(device)

    trainset, testset, num_classes = getattr(dataloader, dataset_name)(**dataset_args)

    valid_state = valid_iter(net, testset, criterion, 0, device, return_result=True)
    return valid_state
