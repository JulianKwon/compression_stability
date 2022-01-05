import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_dim=8, output_len=1, paddings=(0,) * 3, output_paddings=(1, 1, 0), strides=(2, 2, 2),
                 kernel_sizes=(5,) * 3, dilations=(1, 2, 4), embedding_dim=64):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.output_len = output_len
        self.hidden_dim = [self.input_dim, 32, 64, 64]
        self.paddings = paddings
        self.output_paddings = output_paddings
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.embedding_dim = embedding_dim
        self.enc_l_out = lambda l_in, padding, kernel_size, stride, dilation: int(
            (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

        self.dec_l_out = lambda l_in, padding, output_padding, kernel_size, stride, dilation: int(
            (l_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1)

        self.l_in = 7 * 24

        self.inp = torch.randn(100, 8, self.l_in)  # n, c_in, l_in

        # Encoder

        self.encoder = nn.Sequential(
            nn.Conv1d(66, 32, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(66, 32, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(66, 32, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(),
        )

        enc_convs = nn.Sequential()
        for i in range(3):
            enc_convs.add_module('conv%d' % (i + 1),
                                 nn.Conv1d(self.hidden_dim[i], self.hidden_dim[i + 1], self.kernel_sizes[i],
                                           self.strides[i], self.paddings[i],
                                           self.dilations[i]))
            enc_convs.add_module('relu%d' % (i + 1), nn.LeakyReLU())
            self.l_in = self.enc_l_out(self.l_in, self.paddings[i], self.kernel_sizes[i], self.strides[i],
                                       self.dilations[i])

        self.fc_in = self.l_in * self.hidden_dim[-1]
        # self.fc_dims = (self.fc_in, self.embedding_dim * 2)

        self.fc1 = nn.Linear(self.fc_in, self.embedding_dim)
        self.fc2 = nn.Linear(self.fc_in, self.embedding_dim)
        self.fc3 = nn.Linear(self.embedding_dim, self.fc_in)

        # enc_fcs = nn.Sequential()
        # enc_fcs.add_module('fc%d' % (i + 1), nn.Linear(self.fc_dims[i], self.fc_dims[i + 1]))

        self.l_out = self.l_in

        # Decoder
        # dec_fcs = nn.Sequential()
        # for i in range(2):
        #     if i == 0:
        #         in_c = self.embedding_dim
        #     else:
        #         in_c = self.fc_dims[-1 - i]
        # dec_fcs.add_module('fc%d' % (i + 1), nn.Linear(self.embedding_dim, self.l_in))

        dec_convs = nn.Sequential()
        for i in range(3):
            if i < 2:
                dec_convs.add_module('conv%d' % (i + 1),
                                     nn.ConvTranspose1d(self.hidden_dim[-1 - i], self.hidden_dim[-2 - i],
                                                        self.kernel_sizes[-1 - i],
                                                        self.strides[-1 - i], self.paddings[-1 - i],
                                                        self.output_paddings[-1 - i],
                                                        dilation=self.dilations[-1 - i]))
            else:
                dec_convs.add_module('conv%d' % (i + 1),
                                     nn.ConvTranspose1d(self.hidden_dim[-1 - i], 1,
                                                        self.kernel_sizes[-1 - i],
                                                        self.strides[-1 - i], self.paddings[-1 - i],
                                                        self.output_paddings[-1 - i],
                                                        dilation=self.dilations[-1 - i]))
            dec_convs.add_module('relu%d' % (i + 1), nn.LeakyReLU())
            self.l_out = self.dec_l_out(self.l_out, self.paddings[i], self.output_paddings[i], self.kernel_sizes[i],
                                        self.strides[i], self.dilations[i])

        self.enc_convs = enc_convs
        # self.enc_fcs = enc_fcs
        # self.dec_fcs = dec_fcs
        self.dec_convs = dec_convs
        # self.final_fc = nn.Linear(66*168, self.output_len)

        self.initialize()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, inp):
        # print(inp.shape)
        out = self.enc_convs(inp)
        # print(out.shape)
        out = out.view(out.shape[0], -1)
        # print(out.shape)
        out, mu, logvar = self.bottleneck(out)
        # print(out.shape)
        # mu, logvar = out[:, :self.embedding_dim], out[:, self.embedding_dim:]
        # print(mu.shape, logvar.shape)
        # out = self.reparameterize(mu, logvar)
        # print(out.shape)
        out = self.fc3(out)
        # print(out.shape)
        out = out.reshape(out.shape[0], int(out.shape[1] / self.l_in), self.l_in)
        # print(out.shape)
        out = self.dec_convs(out)
        # print(out.shape)
        # out = out.view(out.shape[0], -1)
        # out = self.final_fc(out)
        # print(out.shape)
        return out, mu, logvar

# class VAE(nn.Module):
#     def __init__(self, input_dim=8, hidden_dim=[32, 64, 32]):
#         super(VAE, self).__init__()
#
#         input_dim = 8
#         hidden_dim = [input_dim, 32, 64, 32]
#         paddings = (0,) * 3
#         strides = (2, 2, 2)
#         kernel_sizes = (5,) * 3
#         dilations = (1, 2, 4)
#
#         enc_l_out = lambda l_in, padding, kernel_size, stride, dilation: int(
#             (l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
#
#         dec_l_out = lambda l_in, padding, kernel_size, stride, dilation: int(
#             (l_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1)
#
#         l_in = 7 * 24
#
#         inp = torch.randn(100, 8, l_in)  # n, c_in, l_in
#
#         enc_convs = nn.Sequential()
#         for i in range(3):
#             enc_convs.add_module('conv%d' % (i + 1),
#                                  nn.Conv1d(hidden_dim[i], hidden_dim[i + 1], kernel_sizes[i], strides[i], paddings[i],
#                                            dilations[i]))
#             enc_convs.add_module('relu%d' % (i + 1), nn.LeakyReLU())
#             l_in = enc_l_out(l_in, paddings[i], kernel_sizes[i], strides[i], dilations[i])
#
#         fc_in = l_in * hidden_dim[-1]
#         fc_dims = (fc_in, 256, 256)
#
#         enc_fcs = nn.Sequential()
#         for i in range(2):
#             enc_fcs.add_module('fc%d' % (i + 1), nn.Linear(fc_dims[i], fc_dims[i + 1]))
#
#         dec_fcs = nn.Sequential()
#         for i in range(2):
#             dec_fcs.add_module('fc%d' % (i + 1), nn.Linear(fc_dims[-1 - i], fc_dims[-2 - i]))
#
#         dec_convs = nn.Sequential()
#         for i in range(3):
#             dec_convs.add_module('conv%d' % (i + 1),
#                                  nn.ConvTranspose1d(hidden_dim[-1 - i], hidden_dim[-2 - i], kernel_sizes[-1 - i],
#                                                     strides[-1 - i],
#                                                     paddings[-1 - i], dilations[-1 - i]))
#             dec_convs.add_module('relu%d' % (i + 1), nn.LeakyReLU())
#             l_in = dec_l_out(l_in, paddings[i], kernel_sizes[i], strides[i], dilations[i])
#
#
# class VariationalEncoder(nn.Module):
#     def __init__(self, input_dim, fc_dropout, paddings=(0,) * 3, strides=(1,) * 3, kernel_sizes=(5,) * 3,
#                  var_activation=None):
#         super(VariationalEncoder, self).__init__()
#
#         self.input_dim = input_dim
#         self.var_activation = var_activation
#
#         self.convs = nn.Sequential(
#             nn.Conv1d(input_dim, 32, kernel_sizes[0], strides[0], paddings[0]),
#             nn.LeakyReLU(),
#             nn.Conv1d(32, 64, kernel_sizes[1], strides[1], paddings[1]),
#             nn.LeakyReLU(),
#             nn.Conv1d(64, 128, kernel_sizes[2], strides[2], paddings[2]),
#             nn.LeakyReLU(),
#         )
#         self.fc = nn.Linear(512, 256)
#         self.fc_drop = nn.Dropout(p=fc_dropout)
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, input: Tensor, **kwargs: any):
#         output = self.convs(input)
#         output = output.view(input.shape[0], 512)
#         output = self.fc(output)
#         output = self.fc_drop(output)
#         z_mu = output[:, :128]
#         if self.var_activation:
#             z_var = F.softplus(output[:, 128:])
#         else:
#             z_var = output[:, 128:]
#         return [z_mu, z_var]
#
#
# class VariationalDecoder(nn.Module):
#     def __init__(self, traffic_dim, kernel_sizes, strides, paddings, dropout, fc_dropout):
#         super(VariationalDecoder, self).__init__()
#         self.traffic_dim = traffic_dim
#         self.fc = nn.Linear(128 + traffic_dim, 256)
#         self.fc_drop = nn.Dropout(p=fc_dropout)
#         self.convs = nn.Sequential(
#             nn.ConvTranspose1d(64, 32, kernel_sizes[0], strides[0], paddings[0]),
#             nn.LeakyReLU(),
#             nn.Dropout(p=dropout),
#             nn.ConvTranspose1d(32, 128, kernel_sizes[1], strides[1], paddings[1], output_padding=1),
#             nn.LeakyReLU(),
#             nn.Dropout(p=dropout),
#             nn.ConvTranspose1d(128, 128, kernel_sizes[2], strides[2], paddings[2], output_padding=1),
#             nn.LeakyReLU(),
#             nn.Dropout(p=dropout),
#             nn.ConvTranspose1d(128, 1, kernel_sizes[3], strides[3], paddings[3]),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#         )
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, input: Tensor, **kwargs: any):
#         output = F.leaky_relu(self.fc(input))
#         output = self.fc_drop(output)
#         output = output.reshape((input.shape[0], 64, 4))
#         output = self.convs(output)
#         return output.squeeze().T
#
#
# class VAE(nn.Module):
#     def __init__(self, E_kernel_sizes, E_strides, E_paddings, D_kernel_sizes, D_strides, D_paddings, E_fcdropout=0.5,
#                  D_dropout=0.5, D_fcdropout=0.2, var_activation=True, input_dim=13, traffic_dim=5):
#         super(VAE, self).__init__()
#         self.encoder = VariationalEncoder(input_dim, E_kernel_sizes, E_strides, E_paddings, E_fcdropout, var_activation)
#         self.decoder = VariationalDecoder(traffic_dim, D_kernel_sizes, D_strides, D_paddings, D_dropout, D_fcdropout)
#
#     def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
#         """
#         :param mu: (Tensor) Mean of the latent Gaussian
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian
#         :return: Latent vector z
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu
#
#     def forward(self, input: Tensor, traffic_sig: Tensor, **kwargs: any):
#         mu, log_var = self.encoder(input)
#         z = self.reparameterize(mu, log_var)
#         output = torch.cat((z, traffic_sig), dim=1)
#         output = self.decoder(output)
#         return (output, input, mu, log_var)
#
#     def loss_function_rev(self, y, net, *args, **kwargs):
#         recons = args[0]
#         input = args[1]
#         mu = args[2]
#         log_var = args[3]
#
#         kld_weights = kwargs['M_N']
#         recons_loss = F.mse_loss(recons.T, y)
#         kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
#         loss = -recons_loss - kld_weights * kld_loss
#
#         lp_loss = torch.sum(
#             torch.Tensor([torch.norm(param) for name, param in net.named_parameters() if not 'bias' in name]))
#         var_num_total = torch.sum(
#             torch.Tensor([np.prod(param.shape) for name, param in net.named_parameters() if not 'bias' in name]))
#
#         loss = loss + 2 * lp_loss * 10000.0 / var_num_total
#         return loss, recons_loss
#
#     def loss_function(self, y, net, *args, **kwargs):
#         recons = args[0]
#         input = args[1]
#         mu = args[2]
#         log_var = args[3]
#
#         kld_weights = kwargs['M_N']
#         recons_loss = F.mse_loss(recons.T, y)
#         kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
#         loss = recons_loss + kld_weights * kld_loss
#
#         lp_loss = torch.sum(
#             torch.Tensor([torch.norm(param) for name, param in net.named_parameters() if not 'bias' in name]))
#         var_num_total = torch.sum(
#             torch.Tensor([np.prod(param.shape) for name, param in net.named_parameters() if not 'bias' in name]))
#
#         loss = loss + 2 * lp_loss * 10000.0 / var_num_total
#         return loss, recons_loss
