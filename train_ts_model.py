import argparse
import math

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch import nn, optim

from models import resnet18, resnet34, resnet50, LeNet1D

from data import train_test_loader
from training import iteration
from utils import save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch ResNet18 Training')
parser.add_argument('-c', '--cuda-num', default=0, type=int, help='cuda gpu number')
parser.add_argument('-e', '--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('-m', '--mode', default=0, type=int, help='training mode. (0: pretrain, 1: sparse coding)')
parser.add_argument('-r', '--resnet-mode', default='resnet18', type=str, help='resnet mode. resnet18~resnet152')
parser.add_argument('-l', '--lr', default=0.001, type=float, help='learning rate of optimizer')
parser.add_argument('-d', '--dataset', default='mnist', type=str, help='learning rate of optimizer')
parser.add_argument('-n', '--num-class', default=10, type=int, help='number of classes')
parser.add_argument('-p', '--pre-trained', default=False, type=bool, help='model pretrained flag')


def evaluation(labels, preds, scale, mean):
    mse_list = []
    mae_list = []
    r2_list = []
    for i in range(25):
        mse_list.append(math.sqrt(
            mean_squared_error(labels[:, i] * scale + mean, preds[:, i] * scale + mean, multioutput='uniform_average')))
        mae_list.append(
            mean_absolute_error(labels[:, i] * scale + mean, preds[:, i] * scale + mean, multioutput='uniform_average'))
        r2_list.append(r2_score(labels[:, i] * scale + mean, preds[:, i] * scale + mean, multioutput='uniform_average'))

    return mse_list, mae_list, r2_list


def preTrain(net, name, EPOCH, device=3, batch_size=200, label_index=3, early_stop=5, lr=0.001):
    train_loader, test_loader, scaler = train_test_loader(batch_size=batch_size, label_index=label_index)
    net.to('cuda:%d' % device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    best_valid_loss = 999999
    isBest = False

    best_idx = 0

    train_loss = []
    valid_loss = []

    for epoch in range(EPOCH):
        train_state = iteration(net, train_loader, criterion, name, isTrain=True, isConstrain=False, print_freq=100,
                                optimizer=optimizer, epoch=epoch, device=device, cls=False, transpose=True)
        valid_state = iteration(net, test_loader, criterion, name, isTrain=False, isConstrain=False, print_freq=100,
                                optimizer=optimizer, epoch=epoch, device=device, cls=False, transpose=True)

        train_loss.append(train_state['loss'])
        valid_loss.append(valid_state['loss'])

        if best_valid_loss > valid_state['loss']:
            best_valid_loss = valid_state['loss']
            isBest = True
            best_idx = 0

        save_checkpoint(train_state, name, isBest, filename='checkpoint_train')
        save_checkpoint(valid_state, name, isBest, filename='checkpoint_valid')

        isBest = False

        best_idx += 1

        if best_idx > early_stop:
            print('early stopping. Training exits.')
            break

    return train_loss, valid_loss


if __name__ == '__main__':
    args = parser.parse_args()
    net = LeNet1D(5, 100, 25)
    train_loss, test_loss = preTrain(net, 'ev_lenet', 100)