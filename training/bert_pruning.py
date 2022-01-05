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


# def get_residual_idxs(before_layer, after_layer, sparsity=0.5):
#     with torch.no_grad():
#         if isinstance(before_layer, nn.Linear): # bert input fc
#             p = before_layer.weight
#             pa = after_layer.attn.fc.weight
#         elif isinstance(before_layer, nn.LayerNorm):
#             p = before_layer.weight
#             p = p.unsqueeze(1)
#             pa = after_layer.weight
#         else:
#             p = before_layer.norm2.weight
#             p = p.unsqueeze(1)
#             pa = after_layer.attn.fc.weight

#         p = torch.linalg.norm(torch.cat((p , pa), dim=1), dim=1)
#         lambda_ = p.sort()[0][int(sparsity * p.shape[0])]
#         idxs = torch.ones_like(p)
#         idxs[p.abs() < lambda_] = 0
#     return idxs, lambda_

def get_residual_idx(net, sparsity):
    with torch.no_grad():
        p = net.enc_input_fc2.weight.data
        for i, layer in enumerate(net.encs):
            for head in layer.attn.heads:
                p = torch.cat((p, head.value.fc1.weight.data.T), dim=1)
                p = torch.cat((p, head.key.fc1.weight.data.T), dim=1)
                p = torch.cat((p, head.query.fc1.weight.data.T), dim=1)

            p = torch.cat((p, layer.attn.fc.weight), dim=1)
        #             p = torch.cat((p, layer.norm1.weight.data.unsqueeze(1)), dim=1)
        #             p = torch.cat((p, layer.norm2.weight.data.unsqueeze(1)), dim=1)
        p = torch.linalg.norm(p, dim=1)
        lambda_ = p.sort()[0][int(sparsity * p.shape[0])]
        idxs = torch.ones_like(p)
        idxs[p.abs() < lambda_] = 0
    return idxs, lambda_


def fc_prun(layer, idxs, dim='in'):
    assert dim in ['in', 'out']

    with torch.no_grad():
        if dim == 'out':
            layer.weight[idxs == 0] = 0
        elif dim == 'in':
            layer.weight[:, idxs == 0] = 0

        if layer.bias != None and dim == 'out':
            layer.bias[idxs == 0] = 0


def prune_attnlayer(layer, residual_idx, sparsity):
    """
    :param layer: layer to prune
    :param before_layer: norm2 or fc
    :param idxs: pruned idxs from last layer
    :return: layer idx, layer total idxs, layer total lambdas
    """

    # attention layer prune
    head_idxs = []
    head_lams = []
    for i, head in enumerate(layer.attn.heads):
        v_idx, v_lam = get_prun_idx(head.value.fc1, sparsity)
        k_idx, k_lam = get_prun_idx(head.key.fc1, sparsity)
        q_idx, q_lam = get_prun_idx(head.query.fc1, sparsity)

        fc_prun(head.value.fc1, residual_idx, dim='in')
        fc_prun(head.key.fc1, residual_idx, dim='in')
        fc_prun(head.query.fc1, residual_idx, dim='in')

        fc_prun(head.value.fc1, v_idx, dim='out')
        fc_prun(head.key.fc1, k_idx, dim='out')
        fc_prun(head.query.fc1, q_idx, dim='out')

        if i > 0:
            heads_fc_idx = torch.cat((heads_fc_idx, v_idx))
        else:
            heads_fc_idx = v_idx

        head_idxs.append({'v': v_idx, 'k': k_idx, 'q': q_idx})
        head_lams.append({'v': v_lam, 'k': k_lam, 'q': q_lam})

    # prune attention out fc layer
    fc_prun(layer.attn.fc, heads_fc_idx, dim='in')
    fc_prun(layer.attn.fc, residual_idx, dim='out')

    # fc layers prune
    layer_fc2_idx, fc2_lam = get_prun_idx(layer.fc2, sparsity)

    fc_prun(layer.norm1, residual_idx, dim='out')

    fc_prun(layer.fc2, residual_idx, dim='in')
    fc_prun(layer.fc2, layer_fc2_idx, dim='out')
    fc_prun(layer.fc1, layer_fc2_idx, dim='in')
    fc_prun(layer.fc1, residual_idx, dim='out')

    fc_prun(layer.norm2, residual_idx, dim='out')

    # idxs
    idxs = {
        'head_idxs': head_idxs,
        'heads_fc_idx': heads_fc_idx,
        'layer_fc2_idx': layer_fc2_idx,
    }

    # lambs
    lams = {
        'head_lams': head_lams,
        'fc2_lam': fc2_lam
    }

    return idxs, lams


def bert_prune(net, sparsity):
    total_idxs = {}
    total_lambs = {}

    residual_idxs, residual_lambdas = get_residual_idx(net, sparsity)
    total_idxs['residual_idx'] = residual_idxs
    total_lambs['residual_lam'] = residual_lambdas

    # idxs, lambda_ = get_prun_idx(net.enc_input_fc)
    fc_prun(net.enc_input_fc2, residual_idxs, dim='out')

    islast = False

    for i, layer in enumerate(net.encs):
        layer_idxs, layer_lambs = prune_attnlayer(layer, residual_idxs, sparsity)
        total_idxs['enc_layer%d' % i] = layer_idxs
        total_lambs['enc_layer%d' % i] = layer_lambs

    fc_prun(net.out_fc, residual_idxs.tile(100), dim='in')
    return total_idxs, total_lambs


def fc_compress(layer, idxs, dim='in'):
    assert dim in ['in', 'out']
    with torch.no_grad():
        if dim == 'out':
            layer.weight = nn.Parameter(layer.weight.data[idxs == 1])
        elif dim == 'in':
            layer.weight = nn.Parameter(layer.weight.data[:, idxs == 1])

        if layer.bias != None and dim == 'out':
            layer.bias = nn.Parameter(layer.bias.data[idxs == 1])


def compress_attnlayer(layer, layer_idx, idxs):
    residual_idx = idxs['residual_idx']
    layer_idxs = idxs['enc_layer%d' % layer_idx]
    # attention layer prune
    for i, head in enumerate(layer.attn.heads):
        fc_compress(head.value.fc1, residual_idx, dim='in')
        fc_compress(head.key.fc1, residual_idx, dim='in')
        fc_compress(head.query.fc1, residual_idx, dim='in')

        fc_compress(head.value.fc1, layer_idxs['head_idxs'][i]['v'], dim='out')
        fc_compress(head.key.fc1, layer_idxs['head_idxs'][i]['k'], dim='out')
        fc_compress(head.query.fc1, layer_idxs['head_idxs'][i]['q'], dim='out')

    # prune attention out fc layer
    fc_compress(layer.attn.fc, layer_idxs['heads_fc_idx'], dim='in')
    fc_compress(layer.attn.fc, residual_idx, dim='out')

    fc_compress(layer.norm1, residual_idx, dim='out')

    fc_compress(layer.fc2, residual_idx, dim='in')
    fc_compress(layer.fc2, layer_idxs['layer_fc2_idx'], dim='out')
    fc_compress(layer.fc1, layer_idxs['layer_fc2_idx'], dim='in')
    fc_compress(layer.fc1, residual_idx, dim='out')

    fc_compress(layer.norm2, residual_idx, dim='out')

    norm1_w = layer.norm1.weight.data
    norm1_b = layer.norm1.bias.data
    norm2_w = layer.norm2.weight.data
    norm2_b = layer.norm2.bias.data

    layer.norm1 = nn.LayerNorm(len(norm1_w))
    layer.norm2 = nn.LayerNorm(len(norm2_w))

    layer.norm1.weight = nn.Parameter(norm1_w)
    layer.norm2.weight = nn.Parameter(norm2_w)
    layer.norm1.bias = nn.Parameter(norm1_b)
    layer.norm2.bias = nn.Parameter(norm2_b)


def compress_bert(net, idxs):
    fc_compress(net.enc_input_fc2, idxs['residual_idx'], dim='out')

    for i, layer in enumerate(net.encs):
        compress_attnlayer(layer, i, idxs)

    fc_compress(net.out_fc, idxs['residual_idx'].tile(100), dim='in')
#     net.pos = PositionalEncoding(idxs['residual_idx'][idxs['residual_idx'] == 1].size(0))
