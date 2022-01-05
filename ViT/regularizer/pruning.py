#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/08/20 1:31 오후
# @Author    : Junhyung Kwon
# @Site      :
# @File      : bert_pruning.py
# @Software  : PyCharm

import torch
import torch.nn as nn


def get_prun_idx(layer, sparsity=0.5):
    idxs = None
    lambda_ = None

    with torch.no_grad():
        p = layer.weight.data
        p = torch.linalg.norm(p, dim=1)
        lambda_ = p.sort()[0][int(sparsity * p.shape[0])]
        idxs = torch.ones_like(p)
        idxs[p.abs() < lambda_] = 0
    return idxs, lambda_


def attn_prun_idx(proj, sparsity=0.5, n_heads=12):
    idxs = None
    lambda_ = None

    with torch.no_grad():
        p = proj.weight.data
        dims = p.shape[0]
        p = torch.linalg.norm(p, dim=1)
        idxs = torch.ones((dims))
        lams = []
        for i in range(n_heads):
            head = p[i * (dims // n_heads):i * (dims // n_heads) + (dims // n_heads)]
            lam = head.sort().values[int(sparsity * head.shape[0])]
            indices = head.sort().indices[:int(sparsity * head.shape[0])]
            idxs[indices + i * (dims // n_heads)] = 0
            lams.append(lam)
    return idxs, lams


def conv_prun_idx(layer, sparsity=0.5, structured=True):
    with torch.no_grad():
        if structured:
            p = layer.weight.view(layer.weight.shape[0], -1)
            p = torch.linalg.norm(p, dim=1)
            lambda_ = p.sort()[0][int(sparsity * p.shape[0])]
            conv_idxs = torch.ones_like(p)
            conv_idxs[p.abs() < lambda_] = 0
        else:
            p = layer.weight.abs()
            lambda_ = p.view(-1).sort()[0][int(sparsity * p.view(-1).shape[0])]
            conv_idxs = torch.ones_like(p)
            conv_idxs[p <= lambda_] = 0

    return conv_idxs, lambda_


def fc_prun(layer, idxs, dim='in'):
    assert dim in ['in', 'out']

    with torch.no_grad():
        if dim == 'out':
            layer.weight[idxs == 0] = 0
        elif dim == 'in':
            layer.weight[:, idxs == 0] = 0

        if layer.bias != None and dim == 'out':
            layer.bias[idxs == 0] = 0


def pos_prun(layer, idxs):
    with torch.no_grad():
        layer.pos_embedding[:, :, idxs == 0] = 0


def prune_vit(net, ratio):
    total_idxs = {}
    total_lambdas = {}

    # embedding pruning
    idxs, lambda_ = conv_prun_idx(net.patch_embedding, sparsity=ratio)
    total_idxs['embedding'] = idxs
    total_lambdas['embedding'] = lambda_

    fc_prun(net.patch_embedding, idxs, dim='out')
    pos_prun(net.positional_embedding, idxs)

    total_idxs['blocks'] = []
    total_lambdas['blocks'] = []

    # idxs dim 2
    for layer in net.transformer.blocks:
        block_idxs = {}
        block_lambdas = {}

        # layer norm 1
        fc_prun(layer.norm1, idxs, dim='out')

        # attn
        fc_prun(layer.attn.proj_q, idxs, dim='in')
        fc_prun(layer.attn.proj_k, idxs, dim='in')
        fc_prun(layer.attn.proj_v, idxs, dim='in')
        block_idxs['proj_in'] = idxs

        v_idx, v_lam = attn_prun_idx(layer.attn.proj_v, sparsity=ratio, n_heads=layer.attn.n_heads)
        block_idxs['proj_v'] = v_idx

        fc_prun(layer.attn.proj_v, v_idx, dim='out')

        # projection
        fc_prun(layer.proj, v_idx, dim='in')
        fc_prun(layer.proj, idxs, dim='out')

        # pwff
        pwff_idxs, pwff_lam = get_prun_idx(layer.pwff.fc1, sparsity=ratio)

        fc_prun(layer.pwff.fc1, pwff_idxs, dim='out')
        fc_prun(layer.pwff.fc2, pwff_idxs, dim='in')

        block_idxs['pwff_out'] = pwff_idxs
        block_lambdas['pwff_out'] = pwff_lam
        fc_prun(layer.pwff.fc2, idxs, dim='out')

        # layer norm 2
        fc_prun(layer.norm2, idxs, dim='out')

        total_idxs['blocks'].append(block_idxs)
        total_lambdas['blocks'].append(block_lambdas)

    fc_prun(net.fc, idxs, dim='in')

    return total_idxs, total_lambdas