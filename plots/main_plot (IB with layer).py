import pickle
import argparse
from functools import reduce

import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from model import *

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

rootpath = "/slstore/tianfeng/IB_BNN/figures/CIFAR-10_vgg/baseline/new_seed42_l0.01_epoch200_datasetCIFAR-10_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/"

valset = datasets.CIFAR10(root='/slstore/tianfeng/data', train=False, download=True, transform=transform_test)

valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, pin_memory=True,
                                        num_workers=0)
num_classes = 10

net = VGG16VIWrapper(N=50000, num_classes=num_classes)
net.load(rootpath + "vgg.pt")

conv_layers = []
IBs = []
for layer in net.model.features:
    if layer._get_name() == 'RandConv2d':
        conv_layers.append(layer)
        IBs.append([])


for it, (x, _) in enumerate(valloader):
    x_cuda = x.cuda()
    _ = net.model(x_cuda)
    for i, layer in enumerate(conv_layers):
        IB = layer.kl_output()
        IBs[i].append(IB.item())

IB_mean = [np.mean(i) for i in IBs]



