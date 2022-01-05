# import argparse
#
# import numpy as np
# import torch
# from torch import nn, optim
#
# from data import imagenet
# from models import model_loader
# from training.iteration import iteration_others
# from utils import save_checkpoint, save_data
#
# parser = argparse.ArgumentParser(description='PyTorch ResNet18 Training')
# parser.add_argument('-c', '--cuda-num', default=0, type=int, help='cuda gpu number')
# parser.add_argument('-e', '--epochs', default=50, type=int, help='number of total epochs to run')
# parser.add_argument('-m', '--mode', default=0, type=int, help='training mode. (0: pretrain, 1: sparse coding)')
# parser.add_argument('-r', '--resnet-mode', default='resnet18', type=str, help='resnet mode. resnet18~resnet152')
# parser.add_argument('-l', '--lr', default=0.001, type=float, help='learning rate of optimizer')
# parser.add_argument('-d', '--dataset', default='mnist', type=str, help='learning rate of optimizer')
# parser.add_argument('-n', '--num-class', default=10, type=int, help='number of classes')
# parser.add_argument('-p', '--pre-trained', default=False, type=bool, help='model pretrained flag')
#
#
# def preTrain(name, dataset_name, train_loader, val_loader, num_classes, EPOCH, device=6, lr=0.001):
#     model = model_loader(model_name=name, num_class=num_classes)
#     model.to('cuda:%d' % device)
#
#     criterion = nn.CrossEntropyLoss().to('cuda:%d' % device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     best_valid_loss = 999999
#     isBest = False
#
#     early_stop = 5
#     best_idx = 0
#
#     train_loss = []
#     valid_loss = []
#
#     for epoch in range(EPOCH):
#         train_state = iteration_others(model, train_loader, criterion, isTrain=True, isConstrain=False,
#                                        optimizer=optimizer, device=device, epoch=epoch, sparsity=0, print_freq=20)
#         valid_state = iteration_others(model, val_loader, criterion, isTrain=False, isConstrain=False, print_freq=20,
#                                        epoch=epoch, device=device, sparsity=0)
#
#         train_loss.append(train_state['loss'])
#         valid_loss.append(valid_state['loss'])
#
#         if best_valid_loss > valid_state['loss']:
#             best_valid_loss = valid_state['loss']
#             isBest = True
#             best_idx = 0
#
#         save_checkpoint(train_state, '{}_{}_{}'.format(dataset_name, name, num_classes), isBest, filename='checkpoint_train.pth.tar')
#         save_checkpoint(valid_state, '{}_{}_{}'.format(dataset_name, name, num_classes), isBest, filename='checkpoint_valid.pth.tar')
#
#         isBest = False
#
#         best_idx += 1
#
#         if best_idx > early_stop:
#             print('early stopping. Training exits.')
#             break
#
#
# def sparse_coding(sparsities, dataset_name, train_loader, val_loader, num_classes, EPOCH, device, name='resnet18', preTrained=False):
#     # train_loader, val_loader, num_classes = imagenet(classes=classes)
#
#     criterion = nn.CrossEntropyLoss().to('cuda:%d' % device)
#
#     losses = {}
#
#     for sparsity in sparsities:
#         model = model_loader(model_name=name, pretrain=preTrained, num_class=num_classes)
#         if not preTrained:
#             state = torch.load('runs/%s_%d/model_best.pth.tar' % (name, num_classes))
#             model.load_state_dict(state['state_dict'])
#
#         model.to('cuda:%d' % device)
#
#         optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#         best_valid_loss = 999999
#         isBest = False
#
#         early_stop = 3
#         best_idx = 0
#
#         train_loss = []
#         valid_loss = []
#
#         for epoch in range(EPOCH):
#             if sparsity > 0:
#                 isConstrain = True
#             else:
#                 isConstrain = False
#             train_state = iteration_others(model, train_loader, criterion, isTrain=True, isConstrain=isConstrain,
#                                            optimizer=optimizer, device=device, epoch=epoch, sparsity=sparsity / 100)
#             valid_state = iteration_others(model, val_loader, criterion, isTrain=False, isConstrain=isConstrain,
#                                            print_freq=10000, device=device, epoch=epoch, sparsity=sparsity / 100)
#
#             train_loss.append(train_state['loss'])
#             valid_loss.append(valid_state['loss'])
#
#             if best_valid_loss > valid_state['loss']:
#                 best_valid_loss = valid_state['loss']
#                 isBest = True
#                 best_idx = 0
#
#             save_checkpoint(train_state, '{}_{}_{}/sparsity{}'.format(dataset_name, name, num_classes, sparsity), isBest,
#                             filename='checkpoint_train.pth.tar')
#             save_checkpoint(valid_state, '{}_{}_{}/sparsity{}'.format(dataset_name, name, num_classes, sparsity), isBest,
#                             filename='checkpoint_valid.pth.tar')
#
#             isBest = False
#             best_idx += 1
#
#             if best_idx > early_stop:
#                 print('early stopping. Training exits.')
#                 losses[sparsity] = {
#                     'train': train_loss,
#                     'valid': valid_loss
#                 }
#                 break
#
#
# def inference(sparsity, train_loader, val_loader, num_classes, name, model_name='resnet18', device=0):
#     # train_loader, val_loader, num_classes = imagenet(classes=classes)
#     criterion = nn.CrossEntropyLoss().to('cuda:%d' % device)
#
#     model = model_loader(model_name=model_name)
#     # if sparsity == 0:
#     #     state = torch.load('runs/resnet18/model_best.pth.tar')
#     #     model.load_state_dict(state['state_dict'])
#     # else:
#     state = torch.load('runs/%s_%d/sparsity%d/model_best.pth.tar' % (model_name, num_classes, sparsity))
#     model.load_state_dict(state['state_dict'])
#     model.to('cuda:%d' % device)
#
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     best_valid_loss = 999999
#     isBest = False
#
#     train_state = iteration_others(model, train_loader, criterion, isTrain=False, optimizer=optimizer,
#                                    epoch=0, device=device, sparsity=sparsity / 100)
#     valid_state = iteration_others(model, val_loader, criterion, isTrain=False, print_freq=10000,
#                                    epoch=0, device=device, sparsity=sparsity / 100)
#     train_state = {
#         'epoch': train_state['epoch'],
#         'results': train_state['results'],
#         'loss': train_state['loss'],
#         'softmax_output': train_state['softmax_output'],
#         'labels': train_state['labels'],
#         'preds': train_state['preds']
#     }
#
#     valid_state = {
#         'epoch': valid_state['epoch'],
#         'results': valid_state['results'],
#         'loss': valid_state['loss'],
#         'softmax_output': valid_state['softmax_output'],
#         'labels': valid_state['labels'],
#         'preds': valid_state['preds']
#     }
#
#     save_data(train_state, 'runs/{}_{}_{}/sparsity{}'.format(dataset_name, name, num_classes, sparsity), filename='train.pkl')
#     save_data(valid_state, 'runs/{}_{}_{}/sparsity{}'.format(dataset_name, name, num_classes, sparsity), filename='valid.pkl')
#
#
# if __name__ == '__main__':
#     args = parser.parse_args()
#     print(args)
#
#
#
#     train_loader, val_loader, num_classes = imagenet(classes=args.num_class)
#
#     # sparsity = np.concatenate((np.arange(0, 91, 10), np.arange(91, 100)))
#     sparsity = np.arange(0, 90, 30)
#
#     if args.mode == 0:
#         preTrain(args.resnet_mode, train_loader, val_loader, num_classes, args.epochs, lr=args.lr, device=args.cuda_num,
#                  classes=args.num_class)
#     elif args.mode == 1:
#         sparse_coding(sparsity, train_loader, val_loader, num_classes, args.epochs, name=args.resnet_mode,
#                       device=args.cuda_num, preTrained=args.pre_trained, classes=args.num_class)
#     elif args.mode == 2:
#         for s in sparsity:
#             inference(s, train_loader, val_loader, num_classes, name="%s_sparsecoding_result" % args.resnet_mode,
#                       model_name=args.resnet_mode, classes=args.num_class)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


from data import cifar10
from models import model_loader

from training import off_manifold_attack, attack_score, training, inference

import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse

from tqdm.notebook import tqdm_notebook as tqdm
import pandas as pd

from utils import save_data, load_data, accuracy, plot_grid, save_all, AverageMeter, load_checkpoint

import numpy as np

from copy import deepcopy
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ResNet18 Training')
    parser.add_argument('-c', '--cuda-num', default=0, type=int, help='cuda gpu number')
    parser.add_argument('-e', '--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('-m', '--mode', default=0, type=int, help='training mode. (0: pretrain, 1: sparse coding)')
    parser.add_argument('-r', '--resnet-mode', default='resnet18', type=str, help='resnet mode. resnet18~resnet152')
    parser.add_argument('-l', '--lr', default=0.001, type=float, help='learning rate of optimizer')
    parser.add_argument('-d', '--dataset', default='mnist', type=str, help='learning rate of optimizer')
    parser.add_argument('-n', '--num-class', default=10, type=int, help='number of classes')
    parser.add_argument('-p', '--pre-trained', default=False, type=bool, help='model pretrained flag')