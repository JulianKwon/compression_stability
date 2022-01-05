import torch
from torch import nn, optim

from data import mnist
from models import L0LeNet
from training.trades_train import iteration
from utils import save_checkpoint, save_data

import numpy as np


def preTrain(name, EPOCH):
    device = 5
    train_loader, val_loader, num_classes = mnist()
    model = L0LeNet(10, device=5)

    criterion = nn.CrossEntropyLoss().to('cuda:%d' % device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_valid_loss = 999999
    isBest = False

    early_stop = 3
    best_idx = 0

    train_loss = []
    valid_loss = []

    for epoch in range(EPOCH):
        train_state = iteration(model, train_loader, criterion, isTrain=True, isConstrain=False, optimizer=optimizer,
                                epoch=epoch)
        valid_state = iteration(model, val_loader, criterion, isTrain=False, isConstrain=False, print_freq=10000,
                                epoch=epoch)

        train_loss.append(train_state['loss'])
        valid_loss.append(valid_state['loss'])

        if best_valid_loss > valid_state['loss']:
            best_valid_loss = valid_state['loss']
            isBest = True
            best_idx = 0

        save_checkpoint(train_state, name, isBest, filename='checkpoint_train.pth.tar')
        save_checkpoint(valid_state, name, isBest, filename='checkpoint_valid.pth.tar')

        isBest = False

        best_idx += 1

        if best_idx > early_stop:
            print('early stopping. Training exits.')
            break


def sparse_coding(sparsities, EPOCH, name='l0LeNet_sparsecoding'):
    train_loader, val_loader, num_classes = mnist()

    criterion = nn.CrossEntropyLoss().to('cuda:5')

    for sparsity in sparsities:
        model = L0LeNet(10, device=5, layer_sparsity=(sparsity/100,)*4)
        model.load_state_dict(torch.load('runs/lenet_mnist.pth'))
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_valid_loss = 999999
        isBest = False

        early_stop = 3
        best_idx = 0

        train_loss = []
        valid_loss = []

        for epoch in range(EPOCH):
            train_state = iteration(model, train_loader, criterion, isTrain=True, isConstrain=True, optimizer=optimizer,
                                    epoch=epoch)
            valid_state = iteration(model, val_loader, criterion, isTrain=False, isConstrain=True, print_freq=10000,
                                    epoch=epoch)

            train_loss.append(train_state['loss'])
            valid_loss.append(valid_state['loss'])

            if best_valid_loss > valid_state['loss']:
                best_valid_loss = valid_state['loss']
                isBest = True
                best_idx = 0

            save_checkpoint(train_state, name + '/sparsity%d'%sparsity, isBest, filename='checkpoint_train.pth.tar')
            save_checkpoint(valid_state, name + '/sparsity%d'%sparsity, isBest, filename='checkpoint_valid.pth.tar')

            isBest = False
            best_idx += 1

            if best_idx > early_stop:
                print('early stopping. Training exits.')
                break


def inference(sparsity, name, device_num=3):
    train_loader, val_loader, num_classes = mnist()
    criterion = nn.CrossEntropyLoss().to('cuda:%d'%device_num)

    model = L0LeNet(10, device=device_num)
    if sparsity == 0:
        model.load_state_dict(torch.load('runs/lenet_mnist.pth'))
    else:
        state = torch.load('runs/l0LeNet_sparsecoding2/sparsity%d/model_best.pth.tar' % sparsity)
        model.load_state_dict(state['state_dict'])
    model.to('cuda:%d'%device_num)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_valid_loss = 999999
    isBest = False

    train_state = iteration(model, train_loader, criterion, isTrain=False, isConstrain=True, optimizer=optimizer,
                            epoch=0, device=device_num)
    valid_state = iteration(model, val_loader, criterion, isTrain=False, isConstrain=True, print_freq=10000,
                            epoch=0, device=device_num)
    train_state = {
        'epoch': train_state['epoch'],
        'results': train_state['results'],
        'loss': train_state['loss'],
        'softmax_output': train_state['softmax_output'],
        'labels': train_state['labels'],
        'preds': train_state['preds']
    }

    valid_state = {
        'epoch': valid_state['epoch'],
        'results': valid_state['results'],
        'loss': valid_state['loss'],
        'softmax_output': valid_state['softmax_output'],
        'labels': valid_state['labels'],
        'preds': valid_state['preds']
    }

    save_data(train_state, name + '/sparsity%d'%sparsity, filename='train.pkl')
    save_data(valid_state, name + '/sparsity%d'%sparsity, filename='valid.pkl')



if __name__ == '__main__':
    # preTrain('l0LeNet_pretrain', 100)
    # sparsity = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99)
    sparsity = np.concatenate((np.arange(0, 91, 10), np.arange(91, 100)))
    # sparse_coding(sparsity, 50, name='l0LeNet_sparsecoding2')

    for s in sparsity:
        inference(s, name="l0LeNet_sparsecoding_result")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
