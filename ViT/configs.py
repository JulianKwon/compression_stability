#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/06 7:36 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : configs.py
# @Software  : PyCharm


Basic_args = {
    'cifar10_vitB': {
        'device': 'cuda:5',
        'net_name': 'vit_B_16_imagenet1k',
        'dataset_name': 'cifar10',
        'dataset_args': {
            'batch_size': 128,
        },
        'early_stopping_step': 7,
        'lr': 0.0005,
        'model_args': {
            'num_classes': 10,
            'image_size': 32,
            'in_channels': 3,
        },
        'model_name': 'B_16_imagenet1k',
    },
    'cifar100_vitB': {
        'device': 'cuda:5',
        'net_name': 'vit_B_16_imagenet1k',
        'dataset_name': 'cifar100',
        'dataset_args': {
            'batch_size': 128,
        },
        'early_stopping_step': 7,
        'lr': 0.0005,
        'model_args': {
            'num_classes': 100,
            'image_size': 32,
            'in_channels': 3,
        },
        'model_name': 'B_16_imagenet1k',
    },
    'fmnist_vitB': {
        'device': 'cuda:5',
        'net_name': 'vit_B_16_imagenet1k',
        'dataset_name': 'fmnist',
        'dataset_args': {
            'batch_size': 128,
        },
        'early_stopping_step': 7,
        'lr': 0.0005,
        'model_args': {
            'num_classes': 10,
            'image_size': 28,
            'in_channels': 1,
        },
        'model_name': 'B_16_imagenet1k',
    },
    'mnist_vitB': {
        'device': 'cuda:5',
        'net_name': 'vit_B_16_imagenet1k',
        'dataset_name': 'mnist',
        'dataset_args': {
            'batch_size': 128,
        },
        'early_stopping_step': 7,
        'lr': 0.0005,
        'model_args': {
            'num_classes': 10,
            'image_size': 28,
            'in_channels': 1,
        },
        'model_name': 'B_16_imagenet1k',
    },
    'imagenet10_vitB': {
        'device': 'cuda:5',
        'net_name': 'vit_B_16_imagenet1k',
        'dataset_name': 'imagenet',
        'dataset_args': {
            'batch_size': 16,
            'classes': 10
        },
        'early_stopping_step': 7,
        'lr': 0.0005,
        'model_args': {
            'num_classes': 10,
            'image_size': 224,
            'in_channels': 3,
        },
        'model_name': 'B_16_imagenet1k',
    },

    'cifar10_vitL': {
        'device': 'cuda:5',
        'net_name': 'vit_L_16_imagenet1k',
        'dataset_name': 'cifar10',
        'dataset_args': {
            'batch_size': 128,
        },
        'early_stopping_step': 7,
        'lr': 0.0005,
        'model_args': {
            'num_classes': 10,
            'image_size': 32,
            'in_channels': 3,
        },
        'model_name': 'L_16_imagenet1k',
    },
    'cifar100_vitL': {
        'device': 'cuda:5',
        'net_name': 'vit_L_16_imagenet1k',
        'dataset_name': 'cifar100',
        'dataset_args': {
            'batch_size': 128,
        },
        'early_stopping_step': 7,
        'lr': 0.0005,
        'model_args': {
            'num_classes': 100,
            'image_size': 32,
            'in_channels': 3,
        },
        'model_name': 'L_16_imagenet1k',
    },
    'fmnist_vitL': {
        'device': 'cuda:5',
        'net_name': 'vit_L_16_imagenet1k',
        'dataset_name': 'fmnist',
        'dataset_args': {
            'batch_size': 128,
        },
        'early_stopping_step': 7,
        'lr': 0.0005,
        'model_args': {
            'num_classes': 10,
            'image_size': 28,
            'in_channels': 1,
        },
        'model_name': 'L_16_imagenet1k',
    },
    'mnist_vitL': {
        'device': 'cuda:5',
        'net_name': 'vit_L_16_imagenet1k',
        'dataset_name': 'mnist',
        'dataset_args': {
            'batch_size': 128,
        },
        'early_stopping_step': 7,
        'lr': 0.0005,
        'model_args': {
            'num_classes': 10,
            'image_size': 28,
            'in_channels': 1,
        },
        'model_name': 'L_16_imagenet1k',
    },
    'imagenet10_vitL': {
        'device': 'cuda:5',
        'net_name': 'vit_L_16_imagenet1k',
        'dataset_name': 'imagenet',
        'dataset_args': {
            'batch_size': 16,
            'classes': 10
        },
        'early_stopping_step': 7,
        'lr': 0.0005,
        'model_args': {
            'num_classes': 10,
            'image_size': 224,
            'in_channels': 3,
        },
        'model_name': 'L_16_imagenet1k',
    }
}


