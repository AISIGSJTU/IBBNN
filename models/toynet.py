# https://github.com/1Konny/VIB-pytorch/
# https://github.com/1Konny/VIB-pytorch/blob/dad74f78439dad2eabfe3de506b62c35ed0a35de/model.py#L11
import math
import time
from numbers import Number

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from models.layers.linear import RandLinear


def cuda(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


class ToyNet(nn.Module):

    def __init__(self, N=60000, K=256, n_class=10, mode="vib"):
        super(ToyNet, self).__init__()
        self.mode = mode
        assert mode in ["vib", "bnnvib", "none"]
        self.K = K

        if mode == "vib":
            self.encode = nn.Sequential(
                nn.Linear(784, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 2 * self.K))

            self.decode = nn.Sequential(
                nn.Linear(self.K, n_class))
            self.eps = None
        elif mode == "none":
            self.encode = nn.Sequential(
                nn.Linear(784, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                nn.Linear(1024, self.K))

            self.decode = nn.Sequential(
                nn.Linear(self.K, n_class))
        else:
            self.encode = nn.Sequential(
                nn.Linear(784, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                RandLinear(0.15, N, math.log(0.15), 1024, self.K))

            self.decode = nn.Sequential(
                nn.Linear(self.K, n_class))

    def forward(self, x, num_sample=1):
        if self.mode == "vib":
            if x.dim() > 2: x = x.view(x.size(0), -1)

            statistics = self.encode(x)
            mu = statistics[:, :self.K]
            std = F.softplus(statistics[:, self.K:] - 5, beta=1)
            encoding = self.reparametrize_n(mu, std, num_sample)
            logit = self.decode(encoding)

            if num_sample == 1:
                pass
            elif num_sample > 1:
                logit = F.softmax(logit, dim=2).mean(0)

            info_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
            return logit, 0.1 * info_loss
        elif self.mode == "none":
            if x.dim() > 2: x = x.view(x.size(0), -1)

            statistics = self.encode(x)
            logit = self.decode(statistics)
            return logit, 0
        else:
            if x.dim() > 2: x = x.view(x.size(0), -1)

            statistics, kl = self.encode(x)
            logit = self.decode(statistics)

            info_loss = 0.1 * self.encode[-1].kl_output()
            return logit, kl + info_loss

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        if self.eps is None:
            self.eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))
        return mu + self.eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


def xavier_init(ms):
    for m in ms:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()
