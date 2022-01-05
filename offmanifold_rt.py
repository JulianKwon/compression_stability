#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/13 5:53 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : offmanifold_rt.py
# @Software  : PyCharm

import torch.nn as nn

from training import training

BASE_PATH = '/workspace/paper_works/work_results/finally'
CUDA_NUM = 4

if __name__ == '__main__':
    train_args = {
        'base_path': BASE_PATH,
        'device': 'cuda:%d' % CUDA_NUM,
        'data_type': 'cifar10',
        'schedule': True,
        'data_args': {
            'batch_size': 128,
            'normalization': False
        },
        'model_name': 'resnet34',
        'optim_name': 'Adam',
        'optim_args': {
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 2e-4
        },
        'attack_args': {
            "loss_fn": nn.CrossEntropyLoss(),
            "eps": 8. / 255.,
            "nb_iter": 7,
            "eps_iter": 0.00784,
            "rand_init": True,
            "clip_min": 0.0,
            "clip_max": 1.0,
            "targeted": False
        },
        'loss_name': 'CrossEntropyLoss',
        'epochs': 200,
        'sparsity_ratio': 0.,
        'robust_train': True,
        'pretrain': False,
        'custom_pretrained': False
    }

    training(**train_args)
