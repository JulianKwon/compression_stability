import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from training.regularize import regularize_model, compute_mask
from utils import AverageMeter, reshape_resulting_array, accuracy


def iteration(model, loader, criterion, name='L0LeNet', isTrain=True, isConstrain=False, print_freq=100,
              optimizer=None, epoch=0, device=3, cls=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    total_softmax = []
    total_labels = []
    total_preds = []

    model.train() if isTrain else model.eval()

    for i, (X, y) in enumerate(loader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            X, y = X.to('cuda:%d' % device), y.to('cuda:%d' % device)

        pred = model(X)
        loss = criterion(pred, y)
        losses.update(loss.data, X.size(0))

        if isTrain:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if isConstrain:
            layers = model.layers
            for layer in layers:
                layer.clamp_params()

        # get metric
        pred = F.softmax(pred)
        _, p_data = pred.data.max(dim=1)
        s_data, y_data = pred.data.cpu().detach().numpy(), y.data.cpu().detach().numpy()
        p_data = p_data.cpu().detach().numpy()

        total_softmax.append(s_data)
        total_labels.append(y_data)
        total_preds.append(p_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    total_softmax = reshape_resulting_array(total_softmax)
    total_labels = np.array(total_labels).reshape(-1).squeeze()
    total_preds = np.array(total_preds).reshape(-1).squeeze()

    result = accuracy(total_labels, total_preds, total_softmax)

    # if save_state:
    state = {
        'name': name,
        'epoch': epoch + 1,
        'state_dict': deepcopy(model).cpu().state_dict(),
        'results': result,
        'loss': losses.avg.detach().cpu().numpy(),
        'softmax_output': total_softmax,
        'labels': total_labels,
        'preds': total_preds
    }

    if isTrain:
        state['optimizer'] = deepcopy(optimizer).state_dict()

    return state


def iteration_others(model, loader, criterion, name='resnet18', isTrain=True, isConstrain=False, sparsity=0,
                     print_freq=100, optimizer=None, epoch=0, device=3, cls=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    total_softmax = []
    total_labels = []
    total_preds = []

    model.train() if isTrain else model.eval()

    for i, (X, y) in enumerate(loader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            X, y = X.to('cuda:%d' % device), y.to('cuda:%d' % device)

        pred = model(X)
        loss = criterion(pred, y)
        losses.update(loss.data, X.size(0))

        if isTrain:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if isConstrain and isTrain:
            mask, lamb = compute_mask(model, ratio=sparsity)
            regularize_model(mask, model)

        # get metric
        if cls:
            pred = F.softmax(pred, dim=1)

        _, p_data = pred.data.max(dim=1)
        s_data, y_data = pred.data.cpu().detach().numpy(), y.data.cpu().detach().numpy()
        p_data = p_data.cpu().detach().numpy()

        total_softmax.append(s_data)
        total_labels.append(y_data)
        total_preds.append(p_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            print(' Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    total_softmax = reshape_resulting_array(total_softmax)
    total_labels = np.array(total_labels).reshape(-1).squeeze()
    total_preds = np.array(total_preds).reshape(-1).squeeze()

    print(total_softmax.shape, total_labels.shape, total_preds.shape)

    result = accuracy(total_labels, total_preds, total_softmax)

    # if save_state:
    state = {
        'name': name,
        'epoch': epoch + 1,
        'state_dict': deepcopy(model).cpu().state_dict(),
        'results': result,
        'loss': losses.avg.detach().cpu().numpy(),
        'softmax_output': total_softmax,
        'labels': total_labels,
        'preds': total_preds
    }
    # if isTrain:
    #     state['optimizer'] = deepcopy(optimizer).state_dict()

    return state
