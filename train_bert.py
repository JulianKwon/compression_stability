#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/08/25 7:37 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : train_bert.py
# @Software  : PyCharm

import argparse
import math
import random

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch import nn, optim

from data import ev_data_loader, MetroTrafficDataset
from models.Transformer.BERT import BERT
from training import iteration
from utils import save_checkpoint, save_data

parser = argparse.ArgumentParser(description='PyTorch ResNet18 Training')
parser.add_argument('-c', '--cuda-num', default=0, type=int, help='cuda gpu number')
parser.add_argument('-e', '--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('-m', '--mode', default=0, type=int, help='training mode. (0: pretrain, 1: sparse coding)')
parser.add_argument('-r', '--random-trials', default=50, type=int, help='maximum number of trials for random search')


def evaluation(labels, preds, scale, mean):
    mse_list = []
    mae_list = []
    r2_list = []
    for i in range(labels.shape[1]):
        mse_list.append(math.sqrt(
            mean_squared_error(labels[:, i] * scale + mean, preds[:, i] * scale + mean, multioutput='uniform_average')))
        mae_list.append(
            mean_absolute_error(labels[:, i] * scale + mean, preds[:, i] * scale + mean, multioutput='uniform_average'))
        r2_list.append(r2_score(labels[:, i] * scale + mean, preds[:, i] * scale + mean, multioutput='uniform_average'))

    return mse_list, mae_list, r2_list


def train_regularize(train_loader, test_loader, net, name, EPOCH, ratio, stability_check=False,
                     reg_criterion='standard', rand_per=0.5, device=3, batch_size=200, label_index=3, early_stop=5,
                     lr=0.001):
    print('%s/sparsity%d' % (name, int(ratio * 100)))
    net.to('cuda:%d' % device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    best_valid_loss = 999999
    isBest = False

    best_idx = 0
    best_state = None

    train_loss = []
    valid_loss = []

    if ratio > 0:
        isConstrain = True
    else:
        isConstrain = False
    for epoch in range(EPOCH):
        train_state = iteration(net, train_loader, criterion, name, isTrain=True, isConstrain=isConstrain,
                                print_freq=300, regularizer='structured_resnet_prune',
                                optimizer=optimizer, epoch=epoch, device=device, cls=False, transpose=False,
                                sparsity=ratio, reg_criterion=reg_criterion, rand_per=rand_per, isBert=True,
                                stability_check=stability_check)
        valid_state = iteration(net, test_loader, criterion, name, isTrain=False, isConstrain=False, print_freq=100,
                                regularizer='structured_resnet_prune',
                                optimizer=optimizer, epoch=epoch, device=device, cls=False, transpose=False,
                                reg_criterion=reg_criterion, rand_per=rand_per, isBert=True)

        train_loss.append(train_state['loss'])
        valid_loss.append(valid_state['loss'])

        if best_valid_loss > valid_state['loss']:
            print('best model')
            best_valid_loss = valid_state['loss']
            isBest = True
            best_idx = 0
            best_state = valid_state

        # save_checkpoint(train_state, '%s' % (name, int(ratio * 100)), isBest, filename='checkpoint_train')
        # save_checkpoint(valid_state, '%s' % (name, int(ratio * 100)), isBest, filename='checkpoint_valid')

        isBest = False

        best_idx += 1

        if best_idx > early_stop:
            print('early stopping. Training exits.')
            break

    return train_loss, valid_loss, best_state, best_valid_loss


def randomsearch(train_loader, test_loader, input_feat, input_len, output_len, args):
    losses = []
    states = []
    params = []

    for i in range(args.random_trials):
        random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        params.append(random_params)
        print(random_params['dim'], random_params['dim'], input_feat, input_len, output_len,
                   random_params['num_layers'], random_params['n_heads'])
        net = BERT(random_params['dim'], random_params['dim'], input_feat, input_len, output_len,
                   random_params['num_layers'], random_params['n_heads'])
        train_loss, test_loss, best_state, best_valid_loss = train_regularize(train_loader, test_loader, net,
                                                                              'mt_bert/lr{lr}_dim{dim}_heads{n_heads}_layers{num_layers}'.format_map(
                                                                                  random_params), args.epochs, ratio=0,
                                                                              device=args.cuda_num, early_stop=7,
                                                                              lr=0.0005)

        losses.append(best_valid_loss)
        states.append(best_state)

    best_idx = np.argmin(losses)
    best_params = params[best_idx]
    best_states = states[best_idx]

    print(best_idx)
    print(best_params)

    return best_params, best_states


if __name__ == '__main__':
    args = parser.parse_args()

    param_grid = {
        'lr': list(np.logspace(np.log10(0.0001), np.log10(0.01), base=10, num=10)),
        'dim': list(16 * (2 ** i) for i in range(0, 4)),
        'n_heads': list(range(6, 9)),
        'num_layers': list(range(1, 9))
        # 'lr': [0.01,0.001],
        # 'dim': [16, 32],
        # 'n_heads': [2, 4],
        # 'num_layers': [1, 2]
    }

    random.seed()

    ev_train, ev_test, ev_scaler = ev_data_loader(batch_size=100, label_index=4, lag=100, horizon=50)
    mt = MetroTrafficDataset('Metro_Interstate_Traffic_Volume.csv', '/workspace/Dataset/TSData/', window_size=24 * 7,
                             prediction_length=24 * 2)
    # mt.X_train = np.transpose(mt.X_train, (0, 2, 1))
    # mt.X_test = np.transpose(mt.X_test, (0, 2, 1))
    # mt.X_valid = np.transpose(mt.X_valid, (0, 2, 1))
    mt_train, _, mt_test = mt.make_dataloader(batch_size=100)

    # ev dataset
    input_feat, input_len, output_len = 5, 100, 25
    ev_best_param, ev_best_state = randomsearch(ev_train, ev_test, input_feat, input_len, output_len, args)

    save_data((ev_best_state, ev_best_param), 'runs/ev_bert', 'best_ev.pkl')

    # metro dataset
    input_feat, input_len, output_len = 66, 24 * 7, 24 * 2
    mt_best_param, mt_best_state = randomsearch(mt_train, mt_test, input_feat, input_len, output_len, args)

    save_data((mt_best_state, mt_best_param), 'runs/mt_bert', 'best_mt.pkl')
