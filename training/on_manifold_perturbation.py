#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/22 4:24 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : on_manifold_perturbation.py
# @Software  : PyCharm
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import mnist
from models import LatentClf, resnet18_1d
from simulation.make_data import get_dataloader
from training import OnManifoldPerturbation
from training.vae_train import ClassifierTrainer, VAETrainer
from utils import load_checkpoint, concatenate, AverageMeter


def accuracy(net, test_loader, device):
    net.eval()
    benign_correct = 0
    total_set = 0
    correct_list = None
    correct_label = None
    correct_softmax = None

    for X, y in test_loader:
        X = X.to(device)
        out = net(X)
        out = F.softmax(out, dim=1)

        _, predicted = out.max(1)
        predicted.eq(y.to(device))
        benign_correct += predicted.eq(y.to(device)).sum().item()

        correct_softmax = concatenate(correct_softmax, out[predicted.eq(y.to(device))].detach().cpu().numpy())
        correct_list = concatenate(correct_list, X[predicted.eq(y.to(device))].detach().cpu().numpy())
        correct_label = concatenate(correct_label, y[predicted.eq(y.to(device))].detach().cpu().numpy())
        total_set += X.size(0)

    benign_acc = str(benign_correct / total_set)
    print('accuracy: {0}'.format(benign_acc))
    return benign_acc, correct_list, correct_label, correct_softmax


def on_manifold_perturb(classifier, gan, correct_loader, epsilon=0.01):
    on_adv = OnManifoldPerturbation(classifier, gan, 'cuda:5', eta=epsilon)

    device = 'cuda:5'

    on_adv_Xs = None
    on_ys = None
    on_zs = None
    on_original_zs = None

    attack_succ_perturbed = None
    attack_succ_true = None
    attack_true_label = None
    attack_succ_label = None

    for X, y in correct_loader:
        #     X = X.reshape((X.shape[0], -1))
        z, z_pert, adv_x = on_adv.perturb(X.to(device), y.long().to(device))
        # on_adv_Xs = concatenate(on_adv_Xs, adv_x.detach().cpu().numpy())
        # on_zs = concatenate(on_zs, z_pert.detach().cpu().numpy())
        # on_ys = concatenate(on_ys, y.detach().cpu().numpy())

        f_x = classifier(adv_x)
        f_x = F.softmax(f_x, dim=1)
        _, predicted = f_x.max(1)

        attack_succ_perturbed = concatenate(attack_succ_perturbed,
                                            adv_x[~predicted.eq(y.to(device))].detach().cpu().numpy())
        attack_succ_true = concatenate(attack_succ_true, X[~predicted.eq(y.to(device))].detach().cpu().numpy())
        attack_true_label = concatenate(attack_true_label, y[~predicted.eq(y.to(device))].detach().numpy())
        attack_succ_label = concatenate(attack_succ_label,
                                        predicted[~predicted.eq(y.to(device))].detach().cpu().numpy())

        on_adv_Xs = concatenate(on_adv_Xs, adv_x.detach().cpu().numpy())
        on_zs = concatenate(on_zs, z_pert.detach().cpu().numpy())
        on_original_zs = concatenate(on_original_zs, z.detach().cpu().numpy())
        on_ys = concatenate(on_ys, y.detach().cpu().numpy())

    attack_succ_rate = attack_succ_perturbed.shape[0] / 10000
    print('Success Ratio: ', attack_succ_rate)

    return on_adv_Xs, on_zs, on_ys, on_original_zs, \
           attack_succ_perturbed, attack_succ_true, attack_true_label, attack_succ_label, attack_succ_rate


def filter_perturbation(gan, l_clf, filter_loader, device):
    on_adv = None
    on_adv_label = None

    correct = 0
    gan.encoder.eval()
    l_clf.eval()

    for i, (data, y_hat) in enumerate(filter_loader):
        out, _ = gan.encoder(data.to(device))
        out = l_clf(out.unsqueeze(1).to(device))
        out = F.softmax(out, dim=1)
        _, pred = out.max(1)

        on_adv = concatenate(on_adv, data[pred.eq(y_hat.to(device))].detach().cpu().numpy())
        on_adv_label = concatenate(on_adv_label, y_hat[pred.eq(y_hat.to(device))].detach().cpu().numpy())

    return on_adv, on_adv_label, np.unique(on_adv_label, return_counts=True)


def train_latent_classifier(l_clf, gan, data_loader, test_loader, device):
    gan.eval()

    z_list = None
    z_mean_list = None
    true_label_list = None

    for data, label in data_loader:
        z_mean, z_var = gan.encoder(data.to(device))
        z = gan.reparameterize(z_mean, z_var)
        z_mean_list = concatenate(z_mean_list, z_mean.detach().cpu().numpy())
        z_list = concatenate(z_list, z.detach().cpu().numpy())
        true_label_list = concatenate(true_label_list, label.detach().numpy())

    latent_loader = get_dataloader(z_list, true_label_list, 256)

    test_z_list = None
    test_z_mean_list = None
    test_true_label_list = None

    for data, label in test_loader:
        z_mean, z_var = gan.encoder(data.to(device))
        z = gan.reparameterize(z_mean, z_var)
        test_z_mean_list = concatenate(test_z_mean_list, z_mean.detach().cpu().numpy())
        test_z_list = concatenate(test_z_list, z.detach().cpu().numpy())
        test_true_label_list = concatenate(test_true_label_list, label.detach().numpy())

    latent_loader = get_dataloader(z_mean_list, true_label_list, 256)
    test_latent_loader = get_dataloader(test_z_mean_list, test_true_label_list, 256, shuffle=False)

    optimizer = torch.optim.Adam(l_clf.parameters())
    criterion = nn.CrossEntropyLoss()

    best_acc = -1
    best_l_clf = None

    for epo in range(30):
        l_clf.train()
        lossavg = AverageMeter()
        for latent, y_hat in latent_loader:
            out = l_clf(latent.unsqueeze(1).to(device))
            loss = criterion(out, y_hat.long().to(device))
            lossavg.update(loss.item(), y_hat.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct = 0
        for latent, y_hat in test_latent_loader:
            out = l_clf(latent.unsqueeze(1).to(device))
            out = F.softmax(out, dim=1)
            _, pred = out.max(1)

            correct += pred.eq(y_hat.to(device)).sum().item()

        if best_acc < correct / test_z_list.shape[0]:
            best_acc = correct / test_z_list.shape[0]
            best_l_clf = copy.deepcopy(l_clf)

        print('epoch: %d\tloss: %.4f\ttestacc: %.4f' % (epo, lossavg.avg, correct / test_z_list.shape[0]))


def on_manifold_perturb_main(CUDA_NUM=5, train_latent_clf=False, filter_latent=True, sparsity_ratio=0., epsilon=0.01):
    device = 'cuda:%d' % CUDA_NUM
    data_loader, test_loader, _ = mnist(128)

    state_dict = load_checkpoint(base_path='./simulation/results', dataset_name='mnist', net_name='vae',
                                 is_best=False, filename='chkpoint_epoch%d.pt' % 14)
    trainer = VAETrainer(data_loader)
    trainer.gan.load_state_dict(state_dict)
    gan = trainer.gan
    gan.to(device)

    clf = ClassifierTrainer(5, data_loader, test_loader)
    # clf.best_model.load_state_dict(torch.load('./simulation/base_resnet18.pt'))
    clf.classifier.load_state_dict(torch.load('./simulation/base_resnet18.pt'))

    if sparsity_ratio > 0:
        my_file = Path('./simulation/base_resnet18_sparsity%d.pt' % int(100 * sparsity_ratio))
        if my_file.exists():
            clf_state = torch.load(my_file)
            clf.classifier.load_state_dict(clf_state)
        else:
            clf.EPOCH = 20
            clf.train(regularize=True, ratio=sparsity_ratio)
            torch.save(clf.best_model.cpu().state_dict(), my_file)
    #         save_checkpoint((trainer.gan.state_dict()),
    #                         base_path='./simulation/results', dataset_name='mnist', net_name='vae',
    #                         is_best=False, filename='chkpoint_sparsity%d.pt' % int(100*sparsity_ratio))

    if not clf.best_model == None:
        classifier = clf.best_model
    else:
        classifier = clf.classifier
    classifier.to(device)

    acc, correct_list, correct_label = accuracy(classifier, test_loader, 'cuda:5')
    correct_loader = get_dataloader(correct_list, correct_label, drop_last=False, shuffle=False)  # testset 에 대해 맞춘 경우

    # on manifold perturbation을 가함
    attack_succ_perturbed, attack_succ_true, attack_true_label, attack_succ_label, attack_succ_rate = on_manifold_perturb(
        classifier, gan, correct_loader, epsilon=epsilon)

    if filter_latent:
        l_clf = LatentClf()

        if train_latent_clf:
            l_clf.to(device)
            train_latent_classifier(gan, l_clf, )
        else:
            l_state_dict = torch.load('./simulation/mnist_l_clf.pt')
            l_clf.load_state_dict(l_state_dict)
            l_clf.to(device)

        # for checking if perturbed data leap approximated latent decision boundary
        filter_loader = get_dataloader(attack_succ_perturbed, attack_true_label, shuffle=False)

        on_adv, on_adv_label, (label_arrays, label_counts) = filter_perturbation(gan, l_clf, filter_loader, device)

    return on_adv, on_adv_label, label_arrays, label_counts, on_adv.shape[0] / 10000.


class OnManifoldRobustTraining(object):
    def __init__(self, train_loader, test_loader, device, sparsity_ratio=0., robust_train=False):

        self.device = device
        self.classifierTrainer = ClassifierTrainer(5, train_loader, test_loader)
        self.classifierTrainer.classifier.load_state_dict(torch.load('./simulation/base_resnet18.pt'))

        self.classifier = self.classifierTrainer.classifier

        state_dict = load_checkpoint(base_path='./simulation/results', dataset_name='mnist', net_name='vae',
                                     is_best=False, filename='chkpoint_epoch%d.pt' % 14)
        trainer = VAETrainer(train_loader)
        trainer.gan.load_state_dict(state_dict)
        self.gan = trainer.gan
        self.gan.to(self.device)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sparsity_ratio = sparsity_ratio

        self.l_clf = LatentClf()
        l_state_dict = torch.load('./simulation/mnist_l_clf.pt')
        self.l_clf.load_state_dict(l_state_dict)
        self.l_clf.to(device)


    def perturb(self, X, y, eta=0.3, k=7, alpha=0.0784):
        mu, var = self.gan.encoder(X.to(self.device))
        z = self.gan.reparameterize(mu, var)
        zeta = z.detach().clone()

        for i in range(k):
            zeta.requires_grad_()
            with torch.enable_grad():
                x_tilde_pert = self.gan.decoder(z + zeta)
                pert_logits = self.classifier(x_tilde_pert)
                loss = F.cross_entropy(pert_logits, y.to(self.device), reduction="sum")
            grad = torch.autograd.grad(loss, [zeta])[0]
            z_pert = z.detach() + alpha * torch.sign(grad.detach())
            zeta = torch.clamp(z_pert - z, min=-eta, max=eta)
            x = self.gan.decoder(z + zeta)
            x = torch.clamp(x, 0, 1)

        return z, z_pert, x


    def iteration(self, loader, optimizer, criterion, sparsity_ratio=0., train=True):
        if train:
            self.classifier.train()
        else:
            self.classifier.eval()
        for i, (X, y) in loader:
            X = X.to(self.device)
            out = self.classifier(X)
            adv_X, adv_y = self.make_adv(X, y)

    def on_robust_train(self, regularize=False, ratio=0.):
        self.classifier.to(self.device)
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001, weight_decay=5e-6)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.EPOCH):
            train_loss = self.iteration(epoch, self.train_loader, optimizer, criterion, sparsity_ratio=ratio)
            test_loss = self.iteration(epoch, self.test_loader, optimizer, criterion, train=False)

            print('epoch {0} summary: train_loss: {1}, test_loss: {2}'.format(epoch, train_loss, test_loss))

            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.best_model = copy.deepcopy(self.classifier)

    def on_manifold_perturb(self, loader, epsilon):
        on_adv = OnManifoldPerturbation(self.classifier, self.gan, 'cuda:5', eta=epsilon)

        on_adv_X_notfiltered = None
        on_ys = None
        on_zs = None
        on_original_zs = None

        attack_succ_perturbed = None
        attack_succ_true = None
        attack_true_label = None
        attack_succ_label = None





        for X, y in loader:
            #     X = X.reshape((X.shape[0], -1))
            z, z_pert, adv_x = on_adv.perturb(X.to(self.device), y.long().to(self.device))
            # on_adv_Xs = concatenate(on_adv_Xs, adv_x.detach().cpu().numpy())
            # on_zs = concatenate(on_zs, z_pert.detach().cpu().numpy())
            # on_ys = concatenate(on_ys, y.detach().cpu().numpy())

            output = self.l_clf()
            out = F.softmax(output, dim=1)
            _, pred = out.max(1)

            idx = pred.eq(y.to(self.device))

            on_adv_filtered = adv_x[idx].detach().cpu().numpy()
            on_adv_filtered_label = y[idx].detach().cpu().numpy()

            # f_x = self.classifier(adv_x)
            # f_x = F.softmax(f_x, dim=1)
            # _, predicted = f_x.max(1)

            attack_succ_perturbed = concatenate(attack_succ_perturbed, on_adv_filtered)
            attack_succ_label = concatenate(attack_succ_label, on_adv_filtered_label)
            attack_true_label = concatenate(attack_true_label, y[idx].detach().numpy())

            attack_succ_perturbed = concatenate(attack_succ_perturbed,
                                                adv_x[~predicted.eq(y.to(self.device))].detach().cpu().numpy())
            attack_succ_true = concatenate(attack_succ_true, X[~predicted.eq(y.to(self.device))].detach().cpu().numpy())
            attack_true_label = concatenate(attack_true_label, y[~predicted.eq(y.to(self.device))].detach().numpy())
            attack_succ_label = concatenate(attack_succ_label,
                                            predicted[~predicted.eq(y.to(self.device))].detach().cpu().numpy())

            on_adv_Xs = concatenate(on_adv_Xs, adv_x.detach().cpu().numpy())
            on_zs = concatenate(on_zs, z_pert.detach().cpu().numpy())
            on_original_zs = concatenate(on_original_zs, z.detach().cpu().numpy())
            on_ys = concatenate(on_ys, y.detach().cpu().numpy())

        attack_succ_rate = attack_succ_perturbed.shape[0] / 10000
        print('Success Ratio: ', attack_succ_rate)

        return on_adv_Xs, on_zs, on_ys, on_original_zs, \
               attack_succ_perturbed, attack_succ_true, attack_true_label, attack_succ_label, attack_succ_rate

    def main(self):
        epsilon = 0.01
        sparsity_ratio = 0.99


        benign_acc, correct_list, correct_label, correct_softmax = accuracy(self.classifier, self.test_loader, 'cuda:5')
        correct_loader = get_dataloader(correct_list, correct_label, drop_last=False,
                                        shuffle=False)  # testset 에 대해 맞춘 경우

        # on manifold perturbation을 가함
        attack_succ_perturbed, attack_succ_true, attack_true_label, attack_succ_label, attack_succ_rate = self.on_manifold_perturb(
            correct_loader, epsilon=epsilon)
