#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/08/19 9:12 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : layers.py
# @Software  : PyCharm

from torch import nn
import torch.nn.functional as F
from .attention import MultiHeadAttentionBlock


class EncoderLayer(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1):
        """
        Encoder of transformer
        :param dim_val:
        :param dim_attn:
        :param n_heads:
        """
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)

    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)

        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm2(x + a)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)

    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(a + x)
        a = self.attn2(x, kv=enc)
        x = self.norm2(a + x)
        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm3(x + a)
        return x

