#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/08/20 10:22 오전
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : transformer.py
# @Software  : PyCharm

import torch.nn as nn

from .attention import PositionalEncoding
from .layers import EncoderLayer, DecoderLayer



class Transformer(nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, enc_seq_len, dec_seq_len, out_seq_len, n_decoder_layers=1, n_encoder_layers=1, n_heads=1):
        """
        vanilla Transformer model
        :param dim_val: dimension of input value
        :param dim_attn: dimension of attention output
        :param input_size: input sequence length
        :param dec_seq_len: decoder input sequence length
        :param out_seq_len: output sequence length
        :param n_decoder_layers: number of decoder layers
        :param n_encoder_layers: number of encoder layers
        :param n_heads: number of heads in multihead attention
        """
        super(Transformer, self).__init__()

        # Initiate encoder and Decoder layers
        encs = []
        for i in range(n_encoder_layers):
            encs.append(EncoderLayer(dim_val, dim_attn, n_heads))

        decs = []
        for i in range(n_decoder_layers):
            decs.append(DecoderLayer(dim_val, dim_attn, n_heads))

        self.encs = nn.Sequential(*encs)
        self.decs = nn.Sequential(*decs)

        self.pos = PositionalEncoding(dim_val)

        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)

    def forward(self, x):
        # encoder
        # e = self.encs[0](self.pos(self.enc_input_fc(x)))
        # for enc in self.encs[1:]:
        #     e = enc(e)

        e = self.encs(self.pos(self.enc_input_fc(x)))
        d = self.decs(self.dec_input_fc(x[:, -self.dec_seq_len:]), e)

        x = self.out_fc(d.flatten(start_dim=1))
        return x