#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Created   : 2021/09/01 7:18 오후
# @Author    : Junhyung Kwon
# @Site      : 
# @File      : LSTM.py
# @Software  : PyCharm

from torch import nn
from torch.nn import functional as F
import random
import torch


# input
class Encoder_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0):
        super(Encoder_LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layer = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout) # seq batch feature

    def init_hidden(self, device):
        h_0 = nn.init.kaiming_normal_(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device=device))
        c_0 = nn.init.kaiming_normal_(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device=device))
        self.hn = (h_0, c_0)

    def forward(self, x, training=True):
        self.init_hidden(self.device)



        # rnn_output, self.hn1 = self.layer1(x, self.hn1)
        # rnn_output = self.dropout(rnn_output)
        # rnn_output, self.hn2 = self.layer2(rnn_output, self.hn2)
        # rnn_output = self.dropout(rnn_output)
        # rnn_output, self.hn3 = self.layer3(rnn_output, self.hn3)

        return rnn_output, [self.hn1, self.hn2, self.hn3]


class Decoder_LSTM(nn.Module):
    def __init__(self, hidden_dim, input_dim, batch_size, device, batch_first=False, dropout=0):
        super(Decoder_LSTM, self).__init__()

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim  # [64, 32, 13]

        self.hidden_dim[0] = 1

        self.layer1 = nn.LSTM(input_dim, hidden_dim[1], 1)
        self.layer2 = nn.LSTM(hidden_dim[1], hidden_dim[2], 1)
        self.layer3 = nn.LSTM(hidden_dim[2], hidden_dim[3], 1)

        self.fc = (nn.Linear(hidden_dim[-1], 1))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, hn):
        rnn_output = x

        rnn_output, self.hn[0] = self.layer1(rnn_output, self.hn[0])
        rnn_output = self.dropout(rnn_output)
        rnn_output, self.hn[1] = self.layer2(rnn_output, self.hn[1])
        rnn_output = self.dropout(rnn_output)
        rnn_output, self.hn[2] = self.layer3(rnn_output, self.hn[2])

        output = F.leaky_relu(self.fc(rnn_output))

        return output, self.hn


class Seq2Seq(nn.Module):
    def __init__(self, hidden_dim, batch_size, device, dec_input_dim, dropout=0):
        super().__init__()

        self.encoder = Encoder_LSTM(hidden_dim, batch_size, device, dropout=dropout).to(device)
        self.decoder = Decoder_LSTM(hidden_dim, dec_input_dim, batch_size, dropout=dropout).to(device)
        self.device = device

        self.hidden_dim = hidden_dim

        self.fc = (nn.Linear(hidden_dim[-1], 1))

    def forward(self, X, y, traffic_sig=None, training=True, teacher_forcing_ratio=0):
        batch_size = y.shape[1]
        horizon_length = y.shape[0]

        is_traffic = False

        if traffic_sig != None:
            is_traffic = True
            traffic_sig = torch.from_numpy(traffic_sig.astype('float')).float().to(self.device)

        X = torch.from_numpy(X.astype('float')).float().to(self.device)

        outputs = torch.zeros(horizon_length, batch_size).to(self.device)

        enc_output, hn = self.encoder(X)

        output = F.leaky_relu(self.fc(enc_output[-1]))

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio and training else False

        for t in range(horizon_length):
            if use_teacher_forcing:
                if t == 0:
                    X_out = X[-1, :, 12].unsqueeze(0).unsqueeze(2)
                else:
                    X_out = y[t - 1].unsqueeze(0).unsqueeze(2)
            else:
                if t == 0:
                    X_out = output.unsqueeze(0)
                else:
                    X_out = output

            if is_traffic:
                X_out = torch.cat((X_out, traffic_sig[t].unsqueeze(0).unsqueeze(2)), 2)

            output, hn = self.decoder(X_out, hn)
            outputs[t] = output.squeeze()

        return outputs
