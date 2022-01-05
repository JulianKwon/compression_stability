#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/13 7:57 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : train_vae_gan.py
# @Software  : PyCharm

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/13 2:34 오후
# @Author    : Junhyung Kwon
# @Site      :
# @File      : train_vae_gan.py
# @Software  : PyCharm
import os
import random
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import argparse

import torch.nn.functional as F

from torchvision.utils import save_image
import torchvision.utils as vutils

from data import cifar10
from utils import save_checkpoint, AverageMeter, load_data, load_checkpoint
from models import VaeGan_pert, model_loader

import numpy as np

manualSeed = 999
print('Random Seed: ', manualSeed)

random.seed(manualSeed)
torch.manual_seed(manualSeed)


def as_variable(mixed, grads=False):
    """
    Get a tensor or numpy array as variable.
    :param mixed: input tensor
    :type mixed: torch.Tensor or numpy.ndarray
    :param gpu: gpu or not
    :type gpu: bool
    :param grads: gradients
    :type grads: bool
    :return: variable
    :rtype: torch.autograd.Variable
    """

    assert isinstance(mixed, np.ndarray) or isinstance(mixed,
                                                       torch.Tensor), 'input needs to be numpy.ndarray or torch.Tensor'

    if isinstance(mixed, np.ndarray):
        mixed = torch.from_numpy(mixed)

    return torch.autograd.Variable(mixed, grads)


def latent_loss(output_mu, output_logvar):
    """
    Latent KLD loss.
    :param output_mu: target images
    :type output_mu: torch.autograd.Variable
    :param output_logvar: predicted images
    :type output_logvar: torch.autograd.Variable
    :return: error
    :rtype: torch.autograd.Variable
    """

    return -0.5 * torch.sum(1 + output_logvar - output_mu.pow(2) - output_logvar.exp())


def reconstruction_loss(batch_images, output_images, absolute_error=False):
    """
    Reconstruction loss.
    :param batch_images: original images
    :type batch_images: torch.autograd.Variable
    :param output_images: output images
    :type output_images: torch.autograd.Variable
    :return: error
    :rtype: torch.autograd.Variable
    """

    if absolute_error:
        return torch.sum(torch.abs(batch_images - output_images))
    else:
        return torch.sum(torch.mul(batch_images - output_images, batch_images - output_images))


def reconstruction_error(batch_images, output_images):
    """
    Reconstruction loss.
    :param batch_images: target images
    :type batch_images: torch.autograd.Variable
    :param output_images: predicted images
    :type output_images: torch.autograd.Variable
    :return: error
    :rtype: torch.autograd.Variable
    """

    return torch.mean(torch.mul(batch_images - output_images, batch_images - output_images))


def decoder_loss(output_reconstructed_classes):
    """
    Adversarial loss for decoder.
    :param output_reconstructed_classes: reconstructed predictions
    :type output_reconstructed_classes: torch.autograd.Variable
    :param output_fake_classes: reconstructed predictions
    :type output_fake_classes: torch.autograd.Variable
    :return: error
    :rtype: torch.autograd.Variable
    """

    return - torch.sum(torch.log(torch.nn.functional.sigmoid(output_reconstructed_classes) + 1e-12))


def discriminator_loss(output_real_classes, output_reconstructed_classes):
    """
    Adversarial loss.
    :param output_real_classes: real predictions
    :type output_real_classes: torch.autograd.Variable
    :param output_reconstructed_classes: reconstructed predictions
    :type output_reconstructed_classes: torch.autograd.Variable
    :return: error
    :rtype: torch.autograd.Variable
    """

    return - torch.sum(torch.log(torch.nn.functional.sigmoid(output_real_classes) + 1e-12) + torch.log(
        1 - torch.nn.functional.sigmoid(output_reconstructed_classes) + 1e-12))


def train(epoch, net, loader, device, encoder_optimizer, decoder_optimizer, discriminator_optimizer, beta, gamma, eta,
          equilibrium, margin, optimize_discriminator=False, reg_ratio=0.):
    global latent_loss, reconstruction_loss, decoder_loss, discriminator_loss

    # latent_losses = AverageMeter()
    reconstruction_losses = AverageMeter()
    # reconstruction_errors = AverageMeter()
    encoder_losses = AverageMeter()
    decoder_losses = AverageMeter()
    discriminator_losses = AverageMeter()
    kl_losses = AverageMeter()

    net.train()
    net.encoder.train()
    net.decoder.train()
    if optimize_discriminator:
        net.discriminator.train()
    else:
        net.discriminator.eval()

    for j, (X, y) in enumerate(loader):
        batch_size = len(X)

        X = X.to(device)

        # mu, log_var = net.encoder(X)
        # z = net.reparameterize(mu, log_var)
        # x_tilde = net.decoder(z)
        # real_classes = net.discriminator(X)
        # reconstructed_classes = net.discriminator(x_tilde)

        # lat_loss = latent_loss(mu, log_var)
        # recon_loss = reconstruction_loss(X, x_tilde)
        # dec_loss = decoder_loss(reconstructed_classes)
        # disc_loss = discriminator_loss(real_classes, reconstructed_classes)
        #
        # latent_losses.update(lat_loss.item(), y.size(0))
        # reconstruction_losses.update(recon_loss.item(), y.size(0))
        # decoder_losses.update(dec_loss.item(), y.size(0))
        # discriminator_losses.update(disc_loss.item(), y.size(0))
        #
        # loss = lat_loss + beta * recon_loss + gamma * dec_loss + eta * torch.sum(
        #     torch.abs(log_var))
        # loss.backward(retain_graph=True)
        # encoder_optimizer.step()
        # net.zero_grad()
        # encoder_losses.update(loss.item(), y.size(0))
        #
        # loss = beta * recon_loss + gamma * dec_loss
        # loss.backward(retain_graph=True)
        # net.discriminator.zero_grad()
        # decoder_optimizer.step()
        #
        # loss_discriminator = gamma * disc_loss
        #
        # if optimize_discriminator:
        #     discriminator_optimizer.zero_grad()
        #     loss_discriminator.backward()
        #     discriminator_optimizer.step()
        #
        # e = reconstruction_error(X, x_tilde)
        # reconstruction_errors.update(e.item(), y.size(0))
        #
        # print('[%02d] encoder loss: %.5f | decoder loss: %.5f | discriminator loss: %.5f' % (
        #     epoch, encoder_losses.avg, discriminator_loss.avg, loss_discriminator.item()))

        x_tilde, x_p, disc_class_tilde, disc_class_gen, disc_class, mu, log_var = net(X)

        lat_loss = latent_loss(mu, log_var)
        recon_loss = reconstruction_loss(X, x_tilde)
        dec_loss = decoder_loss(disc_class_tilde)
        disc_loss = discriminator_loss(disc_class, disc_class_tilde)


        loss_encoder = lat_loss + beta * recon_loss + gamma * dec_loss + eta * torch.sum(
            torch.abs(log_var))
        encoder_optimizer.zero_grad()
        loss_decoder = beta * recon_loss + gamma * dec_loss
        decoder_optimizer.zero_grad()
        loss_disc = gamma * disc_loss

        # loss_discriminator = torch.sum(bce_dis_original) + torch.sum(bce_dis_predicted) + torch.sum(bce_dis_sampled)
        # loss_encoder = torch.sum(kl) + beta * torch.sum(reconle) + gamma * torch.sum(bce_dis_predicted) + eta * torch.sum(torch.abs(log_variances))
        # encoder_optimizer.zero_grad()
        # loss_decoder = gamma * torch.sum(reconle) + (1. - gamma) * torch.sum(bce_dis_predicted) # torch.sum(lambda_mse * mse) - (1.0 - lambda_mse) * loss_discriminator
        # # loss_decoder = beta * torch.sum(reconle) + gamma * torch.sum(bce_dis_predicted)
        # # loss_decoder = torch.sum(beta * reconle) - (1.0 - beta) * loss_discriminator
        # decoder_optimizer.zero_grad()

        reconstruction_losses.update(recon_loss.data, y.size(0))
        decoder_losses.update(dec_loss.data, y.size(0))
        discriminator_losses.update(disc_loss.data, y.size(0))
        encoder_losses.update(loss_encoder.data, y.size(0))
        kl_losses.update(lat_loss.data, y.size(0))
        # selectively disable the decoder of the discriminator if they are unbalanced
        train_dis = optimize_discriminator
        train_dec = True

        if train_dis:
            discriminator_optimizer.zero_grad()

        # encoder
        loss_encoder.backward(retain_graph=True)
        net.zero_grad()  # cleanothers, so they are not afflicted by encoder loss

        # decoder
        if train_dec:
            loss_decoder.backward(retain_graph=True)  # [p.grad.data.clamp_(-1,1) for p in net.decoder.parameters()]
            net.discriminator.zero_grad()  # clean the discriminator

        # discriminator
        if train_dis:
            loss_disc.backward()  # [p.grad.data.clamp_(-1,1) for p in net.discriminator.parameters()]

        encoder_optimizer.step()
        if train_dec:
            decoder_optimizer.step()
        if train_dis:
            discriminator_optimizer.step()

        print('[%02d] encoder loss: %.5f | decoder loss: %.5f | discriminator loss: %.5f | kl loss: %.5f | recon loss: %.5f' % (
            epoch, encoder_losses.avg, decoder_losses.avg, discriminator_losses.avg, kl_losses.avg, reconstruction_losses.avg))

    state = {
        'reconstruction_losses': reconstruction_losses,
        'decoder_losses': decoder_losses,
        'encoder_losses': decoder_losses,
        'discriminator_losses': discriminator_losses,
    }

    return state


def test(epoch, net, loader, device):
    net.eval()
    # latent_losses = AverageMeter()
    # reconstruction_losses = AverageMeter()
    # reconstruction_errors = AverageMeter()
    # decoder_losses = AverageMeter()
    # discriminator_losses = AverageMeter()
    # mean = AverageMeter()
    # var = AverageMeter()
    # logvar = AverageMeter()
    #
    # output_images = None
    # pred_images = None
    # pred_codes = None
    #
    # for j, (X, y) in enumerate(loader):
    #     X = as_variable(X, True).to(device)
    #
    #     mu, log_var = net.encoder(X)
    #     z = net.reparameterize(mu, log_var)
    #     x_tilde = net.decoder(z)
    #     real_classes = net.discriminator(X)
    #     reconstructed_classes = net.discriminator(x_tilde)
    #
    #     e = latent_loss(mu, log_var)
    #     latent_losses.update(e, y.size(0))
    #
    #     # Reconstruction loss.
    #     e = reconstruction_loss(X, x_tilde)
    #     reconstruction_losses.update(e, y.size(0))
    #
    #     # Reconstruction error.
    #     e = reconstruction_error(X, x_tilde)
    #     reconstruction_errors.update(e, y.size(0))
    #
    #     e = decoder_loss(x_tilde)
    #     decoder_losses.update(e, y.size(0))
    #
    #     # Adversarial loss.
    #     e = discriminator_loss(real_classes, reconstructed_classes)
    #     discriminator_losses.update(e, y.size(0))
    #
    #     mean.update(torch.mean(mu).item(), y.size(0))
    #     var.update(torch.var(mu).item(), y.size(0))
    #     logvar.update(torch.mean(log_var).item(), y.size(0))
    #
    #     output_images = np.squeeze(np.transpose(x_tilde.cpu().detach().numpy(), (0, 2, 3, 1)))
    #
    # state = {
    #     'latent_losses': latent_losses,
    #     'reconstruction_losses': reconstruction_losses,
    #     'reconstruction_errors': reconstruction_errors,
    #     'decoder_losses': decoder_losses,
    #     'discriminator_losses': discriminator_losses,
    #     'mean': mean,
    #     'var': var,
    #     'logvar': logvar,
    # }
    #
    # return state

    for j, (x, label) in enumerate(loader):

        x = Variable(x, requires_grad=False).float().to(device)

        out = x.data.cpu()
        out = (out + 1) / 2
        save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), './result/original%s.png' % (epoch), nrow=8)

        out = net(x)  # out=x_tilde
        out = out.data.cpu()
        out = (out + 1) / 2
        save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), './result/reconstructed%s.png' % (epoch),
                   nrow=8)

        out = net(None, 100)  ##out=x_p
        out = out.data.cpu()
        out = (out + 1) / 2
        save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), './result/generated%s.png' % (epoch),
                   nrow=8)

        break


def main(classifier, model_args, n_epochs, optim_args, decay_lr, beta, gamma,
         eta, dataset_args, device, base_path, optimize_discriminator, fig_path=''):
    train_loader, test_loader, _ = cifar10(**dataset_args)

    print(len(train_loader.dataset))
    print(len(test_loader.dataset))

    net = VaeGan_pert(classifier, **model_args)
    net.encoder.to(device)
    net.decoder.to(device)
    net.discriminator.to(device)

    optimizer_encoder = Adam(net.encoder.parameters(), lr=0.0003)
    lr_encoder = ExponentialLR(optimizer_encoder, gamma=decay_lr)

    optimizer_decoder = Adam(net.decoder.parameters(), **optim_args)
    lr_decoder = ExponentialLR(optimizer_decoder, gamma=decay_lr)

    optimizer_discriminator = Adam(net.discriminator.parameters(), **optim_args)
    lr_discriminator = ExponentialLR(optimizer_discriminator, gamma=decay_lr)

    equilibrium = 0.68
    margin = 0.35

    for i in range(n_epochs):
        print('Epoch:%s' % (i))
        state = train(i, net, train_loader, device, optimizer_encoder, optimizer_decoder, optimizer_discriminator, beta,
                      gamma, eta, equilibrium, margin, optimize_discriminator)
        # vstate = test(i, net, test_loader, device)

        lr_encoder.step()
        lr_decoder.step()
        lr_discriminator.step()

        # ------------ model checkpoint save --------------- #
        if i % 10 == 0:
            save_checkpoint((net.encoder.state_dict(), net.decoder.state_dict(), net.discriminator.state_dict()),
                            base_path=base_path, dataset_name='cifar10', net_name='vae_gan',
                            is_best=False, filename='chkpoint2_epoch%d.pt' % i)

            if not os.path.exists(fig_path):
                os.mkdir(fig_path)

        for j, (X, y) in enumerate(test_loader):
            net.eval()

            X = Variable(X, requires_grad=False).float().to(device)

            out = X.data.cpu()
            out = (out + 1) / 2
            save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(),
                       '%s/original%s.png' % (fig_path, i),
                       nrow=8)

            out = net(X)  # out=x_tilde
            out = out.data.cpu()
            out = (out + 1) / 2
            save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(),
                       '%s/reconstructed%s.png' % (fig_path, i), nrow=8)

            out = net(None, 100)  ##out=x_p
            out = out.data.cpu()
            out = (out + 1) / 2
            save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(),
                       '%s/generated%s.png' % (fig_path, i),
                       nrow=8)
            break


def get_classifier():
    net = model_loader('resnet34')
    state = load_checkpoint(BASE_PATH, 'cifar10',  'resnet34', True,
                    filename='benign_train_%d' % int(100 * 0))
    net.load_state_dict(state['state_dict'])
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train auto encoder.')

    parser.add_argument('-epoch', default=50, help='EPOCHs',
                        type=int)
    parser.add_argument('-lr', default=0.0005, help='Learning rate',
                        type=float)
    parser.add_argument('-beta', default=1., help='Beta for optimizer',
                        type=float)
    parser.add_argument('-cuda', default=0, help='cuda number',
                        type=int)

    ars = parser.parse_args()

    BASE_PATH = '/workspace/paper_works/work_results/finally'
    CUDA_NUM = ars.cuda

    torch.autograd.set_detect_anomaly(True)

    classifier = get_classifier()
    classifier.to('cuda:%d' % CUDA_NUM)

    args = {
        'n_epochs': ars.epoch,
        'decay_lr': 0.99,
        'classifier': classifier,
        'beta': 2.,
        'gamma': 0.5,
        'eta': 0.0001,
        'model_args': {
            'z_size': 128,
            'recon_level': 3,
            'latent_dim': 4,
            'device': 'cuda:%d' % CUDA_NUM,
        },
        'dataset_args': {
            'batch_size': 128,
            'normalization': False,
            'drop_last': True
        },
        'optim_args': {
            'lr': ars.lr,
            'weight_decay': 0.,
            'betas': (0.5, 0.9)
        },
        'device': 'cuda:%d' % CUDA_NUM,
        'base_path': BASE_PATH,
        'fig_path': '%s/gan_figs2' % BASE_PATH,
        'optimize_discriminator': False
    }

    main(**args)
