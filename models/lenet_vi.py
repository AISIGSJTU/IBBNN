import torch
import torch.nn as nn
from .layers.batchnorm2d import RandBatchNorm2d
from .layers.conv2d import RandConv2d
from .layers.linear import RandLinear



class Lenet_vi(nn.Module):
    def __init__(self, N, sigma_0=0.15, init_s=0.15, nclass=10):
        super(Lenet_vi, self).__init__()
        self.sigma_0 = sigma_0
        self.N = N
        self.init_s = init_s
        self.nclass = nclass
        self.conv_layers = self._make_conv_layers()
        self.fc_layers = self._make_fc_layers()


    def forward(self, x, sample=True, fix=False):
        kl_sum = 0
        out = x.view(x.size(0), 1, 28, 28)
        for l in self.conv_layers:
            if type(l).__name__.startswith("Rand"):
                out, kl = l.forward(out)
                if kl is not None:
                    kl_sum += kl
            else:
                out = l.forward(out)
        out = out.view(out.size(0), -1)
        for l in self.fc_layers:
            if type(l).__name__.startswith("Rand"):
                out, kl = l.forward(out)
                if kl is not None:
                    kl_sum += kl
            else:
                out = l.forward(out)
        return out, kl_sum

    def _make_conv_layers(self):
        layers = [
            RandConv2d(self.sigma_0, self.N, self.init_s, 1, 6, kernel_size=5, padding=2),
            RandBatchNorm2d(self.sigma_0, self.N, self.init_s, 6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            RandConv2d(self.sigma_0, self.N, self.init_s, 6, 16, kernel_size=5, padding=0),
            RandBatchNorm2d(self.sigma_0, self.N, self.init_s, 16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        ]
        return nn.Sequential(*layers)

    def _make_fc_layers(self):
        widths = [5 * 5 * 16, 120, 84, self.nclass]
        layers = []
        for i in range(len(widths) - 1):
            layers.append(RandLinear(self.sigma_0, self.N, self.init_s, widths[i], widths[i + 1]))
            if i != len(widths) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)