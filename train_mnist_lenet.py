#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/27 10:25 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : train_mnist_lenet.py
# @Software  : PyCharm
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attacker import LinfPGDAttack
from data import mnist
from models import LatentClf
from simulation.make_data import get_dataloader
from training import OnManifoldPerturbation
from training.vae_train import ClassifierTrainer, VAETrainer
from utils import load_data, load_checkpoint, concatenate, save_data

# 시드 고정
random_seed = 999

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


def accuracy(net, test_loader, device, epsilon=1.0, alpha=0.01, k=7):
    net.eval()
    benign_correct = 0
    adv_correct = 0
    total_set = 0
    correct_list = None
    correct_label = None
    adv_list = None
    benign_softmax_list = None
    adversarial_softmax_list = None

    adv = LinfPGDAttack(net, epsilon=epsilon, alpha=alpha, k=k)

    for X, y in test_loader:
        X = X.to(device)
        out = net(X)
        out = F.softmax(out, dim=1)
        benign_softmax_list = concatenate(benign_softmax_list, out.detach().cpu().numpy())
        _, predicted = out.max(1)
        idx = predicted.eq(y.to(device))
        benign_correct += predicted.eq(y.to(device)).sum().item()

        correct_list = concatenate(correct_list, X[idx].detach().cpu().numpy())
        correct_label = concatenate(correct_label, y[idx].detach().cpu().numpy())

        adv_x = adv.perturb(X.to(device), y.long().to(device))

        perturbation = adv_x - X
        out = net(adv_x)
        out = F.softmax(out, dim=1)
        adversarial_softmax_list = concatenate(adversarial_softmax_list, out.detach().cpu().numpy())
        _, predicted = out.max(1)
        adv_correct += predicted.eq(y.to(device)).sum().item()

        adv_list = concatenate(adv_list, perturbation.detach().cpu().numpy())

        total_set += X.size(0)

    benign_acc = benign_correct / total_set
    adv_acc = adv_correct / total_set
    print('benign accuracy: {0}\tadversarial accuracy: {1}'.format(benign_acc, adv_acc))
    return benign_acc, adv_acc, correct_list, correct_label, adv_list, benign_softmax_list, adversarial_softmax_list


class Tilda(torch.nn.Module):
    def __init__(self, batch_size):
        super(Tilda, self).__init__()
        z_tilde = np.random.normal(size=(batch_size, 2))
        self.z_tilde = torch.nn.Parameter(torch.from_numpy(z_tilde).float(), requires_grad=True)

    def forward(self, decoder):
        return decoder(self.z_tilde)


def make_perturbed_data(classifier, gan, l_clf, loader, device, epsilon=0.3, k=7, alpha=0.01, filter_=False):
    on_adv = OnManifoldPerturbation(classifier, gan, device, eta=epsilon, k=k, alpha=alpha)

    on_adv_Xs = None
    on_ys = None
    on_zs = None
    on_original_zs = None
    attack_succ_idxs = None
    orig_X = None
    orig_y = None

    print('creating perturbed data..')

    for X, y in loader:
        #     X = X.reshape((X.shape[0], -1))
        z, z_pert, adv_x = on_adv.perturb(X.to(device), y.long().to(device))
        on_adv_Xs = concatenate(on_adv_Xs, adv_x.detach().cpu().numpy())
        on_zs = concatenate(on_zs, z_pert.detach().cpu().numpy())
        on_ys = concatenate(on_ys, y.detach().cpu().numpy())
        on_original_zs = concatenate(on_original_zs, z.detach().cpu().numpy())
        orig_X = concatenate(orig_X, X.detach().numpy())
        orig_y = concatenate(orig_y, y.detach().numpy())

        f_x = classifier(adv_x)
        f_x = F.softmax(f_x, dim=1)
        _, predicted = f_x.max(1)

        idxs = ~predicted.eq(y.to(device))
        attack_succ_idxs = concatenate(attack_succ_idxs, idxs.detach().cpu().numpy())

    # attack_succ_rate = attack_succ_perturbed.shape[0]/10000
    # print(attack_succ_rate)

    adv_loader = get_dataloader(on_adv_Xs, on_ys)
    z_loader = get_dataloader(on_zs, on_ys)

    on_xs = None
    on_ys = None
    on_preds = None

    print('filtering perturbed data..')

    for x_tilde, y in adv_loader:
        out, _ = gan.encoder(x_tilde.to(device))
        out = l_clf(out.unsqueeze(1))
        out = F.softmax(out, dim=1)
        _, pred = out.max(1)

        idx = pred.eq(y.to(device))

        on_adv_filtered = x_tilde[idx].detach().cpu().numpy()
        on_adv_filtered_label = y[idx].detach().cpu().numpy()

        on_xs = concatenate(on_xs, on_adv_filtered)
        on_ys = concatenate(on_ys, on_adv_filtered_label)
        on_preds = concatenate(on_preds, pred.detach().cpu().numpy())

    final_loader = get_dataloader(on_xs, on_ys)

    if filter_:

        filtered_onxs = None
        filtered_onys = None

        print('organizing training data..')

        for x_tilde, y in final_loader:
            out = classifier(x_tilde.to(device))
            f_x = F.softmax(out, dim=1)
            _, predicted = f_x.max(1)
            idxs = ~predicted.eq(y.to(device))

            filtered_onxs = concatenate(filtered_onxs, x_tilde[idxs].detach().numpy())
            filtered_onys = concatenate(filtered_onys, y[idxs].detach().numpy())

        total_X = concatenate(orig_X, on_xs)
        total_y = concatenate(orig_y, on_ys)

        final_loader = get_dataloader(filtered_onxs, filtered_onys)

    return final_loader


def analysis_per_sparsity(model_type, cuda_num, data_type, device, train_type, loader, test_loader, EPOCHs=10,
                          retrain_models=False, onadv_loader=None,
                          sparsity_ratio=0.9, epsilon=1.0, alpha=0.01, k=7):
    assert train_type in ['base', 'robust', 'original_robust',
                          'onoff'], "train type must be one of ['base', 'robust', 'original_robust', 'onoff']"

    clf = ClassifierTrainer(model_type, cuda_num, loader, test_loader)
    regularize = True
    if sparsity_ratio == 0.:
        regularize = False

    if train_type == 'base' and sparsity_ratio == 0:
        my_file = Path('./simulation/%s/base_%s.pt' % (data_type, model_type))
    else:
        my_file = Path('./simulation/%s/%s_train_v3_%s_sparsity%s.pt' % (
            train_type, data_type, model_type, str(sparsity_ratio).replace('.', '_')))

    if my_file.exists() and not retrain_models:
        clf_state = torch.load(my_file)
        clf.classifier.load_state_dict(clf_state)
        classifier = clf.classifier
    else:
        clf.EPOCH = EPOCHs
        if train_type == 'onoff':
            clf.train(regularize=regularize, ratio=sparsity_ratio, onoff=True, onmanifold_loader=onadv_loader)
        elif train_type == 'original_robust':
            clf.train(regularize=regularize, ratio=sparsity_ratio, AT=True, epsilon=epsilon, alpha=alpha, k=k)
        else:
            clf.train(regularize=regularize, ratio=sparsity_ratio)

        classifier = clf.best_model
        torch.save(classifier.cpu().state_dict(), my_file)
    classifier.to(device)

    acc, adv_acc, _, _, advs, benign_softmax_list, adversarial_softmax_list = accuracy(classifier, test_loader, device,
                                                                                       epsilon=epsilon, alpha=alpha,
                                                                                       k=k)
    return acc, adv_acc, advs, benign_softmax_list, adversarial_softmax_list


def run_sparsity(model_type, cuda_num, data_type, sparsity_ratio, device, epsilon=1.0, alpha=0.01, k=7):
    EPOCHs = 10
    data_loader, test_loader, _ = mnist(128)
    ## LOAD ALL
    state_dict = load_checkpoint(base_path='./simulation/results', dataset_name='mnist', net_name='vae',
                                 is_best=False, filename='chkpoint_epoch%d.pt' % 14)
    trainer = VAETrainer(data_loader, cuda_num=2)
    trainer.gan.load_state_dict(state_dict)
    gan = trainer.gan

    l_clf = LatentClf()
    l_clf.load_state_dict(torch.load('./simulation/%s/mnist_l_clf.pt' % data_type))

    retrain_models = True

    datafile = Path('./simulation/%s/onmanifold_%s_%s_loader.pkl' % (data_type, data_type, model_type))
    onmanifold_train_loader = load_data(datafile)

    acc_result = {
        'normal_acc': None,
        'normal_adv_acc': None,
        'on_acc': None,
        'on_adv_acc': None,
        'off_acc': None,
        'off_adv_acc': None,
        'onoff_acc': None,
        'onoff_adv_acc': None
    }

    softmax_result = {
        'normal_benign_softmax': None,
        'on_benign_softmax': None,
        'off_benign_softmax': None,
        'onoff_benign_softmax': None,
        'normal_adv_softmax': None,
        'on_adv_softmax': None,
        'off_adv_softmax': None,
        'onoff_adv_softmax': None,
    }

    adv_data = {
        'normal_adv_lists': None,
        'on_adv_lists': None,
        'off_adv_lists': None,
        'onoff_adv_lists': None
    }

    # normal
    acc_result['normal_acc'], acc_result['normal_adv_acc'], adv_data['normal_adv_lists'], softmax_result['normal_benign_softmax'], \
    softmax_result['normal_adv_softmax'] = analysis_per_sparsity(model_type, cuda_num, data_type, device,
                                                           'base', data_loader, test_loader, EPOCHs=EPOCHs,
                                                           retrain_models=retrain_models, sparsity_ratio=sparsity_ratio)

    # on AT
    acc_result['on_acc'], acc_result['on_adv_acc'], adv_data['on_adv_lists'], softmax_result['on_benign_softmax'], \
    softmax_result['on_adv_softmax'] = analysis_per_sparsity(model_type, cuda_num, data_type, device,
                                                'robust', onmanifold_train_loader, test_loader, EPOCHs=EPOCHs,
                                                retrain_models=retrain_models,
                                                sparsity_ratio=sparsity_ratio)

    # off AT
    acc_result['off_adv_acc'], acc_result['off_adv_acc'], adv_data['off_adv_lists'], softmax_result['off_benign_softmax'], \
    softmax_result['off_adv_softmax'] = analysis_per_sparsity(model_type, cuda_num, data_type, device,
                                                           'original_robust', data_loader, test_loader, EPOCHs=EPOCHs,
                                                           retrain_models=retrain_models,
                                                           sparsity_ratio=sparsity_ratio, epsilon=epsilon, alpha=alpha,
                                                           k=k)

    # on + off AT
    acc_result['onoff_acc'], acc_result['onoff_adv_acc'], adv_data['onoff_adv_lists'], softmax_result['onoff_benign_softmax'], \
    softmax_result['onoff_adv_softmax'] = analysis_per_sparsity(model_type, cuda_num, data_type, device,
                                                           'onoff', data_loader, test_loader, EPOCHs=EPOCHs,
                                                           retrain_models=retrain_models,
                                                           onadv_loader=onmanifold_train_loader,
                                                           sparsity_ratio=sparsity_ratio, epsilon=epsilon, alpha=alpha,
                                                           k=k)


def multi_GPU(param):
    output_dict = {}
    gpu_idx = semaphore.pop()
    device = 'cuda:%d' % gpu_idx if torch.cuda.is_available() else 'cpu'
    print('Process using :', device)
    acc_result, softmax_result, adv_data = run_sparsity(param['model_type'], gpu_idx, param['data_type'],
                                                        param['sparsity_ratio'], device, epsilon=param['epsilon'],
                                                        alpha=param['alpha'], k=param['k'])
    print('Process end :', device)
    semaphore.append(gpu_idx)
    return_dict[param['sparsity_ratio']] = {'acc_result': acc_result, 'softmax_result': softmax_result,
                                            'adv_data': adv_data}


# , 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.93, 0.95, 0.97, 0.99, 0.993, 0.996, 0.999
sparsity_ratio_list = [0, 0.1]

param_list = []
for sparsity_ratio in sparsity_ratio_list:
    param_list.append(
        {'model_type': 'lenet', 'data_type': 'MNIST', 'sparsity_ratio': sparsity_ratio, 'epsilon': 0.3, 'alpha': 0.01,
         'k': 20})

manager = mp.Manager()
semaphore = manager.list([2, 3, 4, 5, 6])
return_dict = manager.dict()

pool = mp.Pool(processes=5)
pool.map(multi_GPU, param_list)
pool.close()
pool.join()

save_data(return_dict, './simulation/MNIST', 'lenet_total_result.pkl')
