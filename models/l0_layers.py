import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair as pair
from torch.nn.parameter import Parameter


class L0Dense(Module):

    def __init__(self, f_in, f_out, bias=True, sparse_ratio=0., reg=False, device='cpu'):
        super(L0Dense, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.bias = bias
        self.weights = Parameter(torch.Tensor(f_in, f_out))
        self.sparse_ratio = sparse_ratio
        self.device = device
        self.reg = reg

        self.percentile = lambda: self.weights.abs().view(-1).sort()[0][
            int(self.sparse_ratio * self.weights.abs().view(-1).shape[0])].detach().cpu().numpy().tolist()

        self.lamb = self.percentile()

        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            self.use_bias = True

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights)
        self.lamb = self.percentile()

    def l0_computation(self):
        self.mask = torch.ones(self.f_in, self.f_out).to(self.device)
        self.lamb = self.percentile()
        self.mask[torch.abs(self.weights) <= self.lamb] = 0
        return self.mask

    def clamp_params(self):
        with torch.no_grad():
            w = self.weights * self.l0_computation()
        self.weights.data = w

    def get_mask(self):
        return self.mask

    def get_lamb(self):
        return self.lamb

    def forward(self, _input):
        xin = _input.mm(self.weights)

        if self.use_bias:
            output = xin.add_(self.bias)

        return output

    def __repr__(self):
        s = ('{name}({f_in} -> {f_out}, sparse_ratio={sparse_ratio}')
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class L0Conv2d(Module):

    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 sparse_ratio=0., reg=False, device='cpu'):

        super(L0Conv2d, self).__init__()

        assert c_in % groups == 0, 'in_channels must be divisible by groups'
        assert c_out % groups == 0, 'out_channels must be divisible by groups'

        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.groups = groups
        self.bias = bias
        self.sparse_ratio = sparse_ratio
        self.device = device
        self.reg = reg

        self.weights = Parameter(torch.Tensor(c_out, c_in // groups, *self.kernel_size))

        self.h_out = lambda h_in: np.floor(
            (h_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        self.w_out = lambda w_in: np.floor(
            (w_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        self.percentile = lambda: self.weights.abs().view(-1).sort()[0][
            int(self.sparse_ratio * self.weights.abs().view(-1).shape[0])].detach().cpu().numpy().tolist()

        if bias:
            self.bias = Parameter(torch.Tensor(c_out))
            self.use_bias = True

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights)
        self.lamb = self.percentile()

    def l0_computation(self):
        self.mask = torch.ones(self.c_out, self.c_in // self.groups, *self.kernel_size, device=self.device)
        self.lamb = self.percentile()
        self.mask[torch.abs(self.weights) <= self.lamb] = 0
        return self.mask

    def clamp_params(self):
        with torch.no_grad():
            w = self.weights * self.l0_computation()
        self.weights.data = w

    def get_mask(self):
        return self.mask

    def get_lamb(self):
        return self.lamb

    def forward(self, _input):
        b = None if not self.use_bias else self.bias
        output = F.conv2d(_input, self.weights, b, self.stride, self.padding, self.dilation, self.groups)

        return output

    def __repr__(self):
        # , temperature={temperature}, prior_prec={prior_prec}, '
        #     'lamba={lamba}, local_rep={local_rep}

        s = ('{name}({c_in}, {c_out}, kernel_size={kernel_size}, stride={stride}, sparse_ratio={sparse_ratio}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
