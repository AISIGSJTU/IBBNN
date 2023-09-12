import torch
import torch.nn as nn
from .layers.batchnorm2d import RandBatchNorm2d
from .layers.conv2d import RandConv2d
from .layers.linear import RandLinear



class MLP_vi(nn.Module):
    def __init__(self, widths: list, N, sigma_0=0.15, init_s=0.15):
        super(MLP_vi, self).__init__()
        self.widths = widths
        self.N = N
        self.sigma_0 = sigma_0
        self.init_s = init_s
        self.layers = self._make_layers(self.widths)


    def forward(self, x):
        kl_sum = 0
        out = x.reshape(x.shape[0], -1)
        for l in self.layers:
            if type(l).__name__.startswith("Rand"):
                out, kl = l.forward(out)
                if kl is not None:
                    kl_sum += kl
            else:
                out = l.forward(out)
        return out, kl_sum

    def _make_layers(self, widths):
        layers = []
        for i in range(len(widths) - 1):
            layers.append(RandLinear(self.sigma_0, self.N, self.init_s, widths[i], widths[i + 1]))
        return nn.Sequential(*layers)

