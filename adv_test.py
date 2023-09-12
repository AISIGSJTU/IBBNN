import pickle
import argparse

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tqdm

from model import *
from attacks.attack import PGD, FGSM

parser = argparse.ArgumentParser(description='adversarial robustness test')
parser.add_argument('-a', '--attack', default='fgsm', type=str, help='Attack method')
args = parser.parse_args()


# class Normalize(nn.Module):
#     def __init__(self, mean, std):
#         super(Normalize, self).__init__()
#         self.register_buffer('mean', torch.Tensor(mean))
#         self.register_buffer('std', torch.Tensor(std))
#
#     def forward(self, input):
#         # Broadcasting
#         mean = self.mean.reshape(1, 3, 1, 1)
#         std = self.std.reshape(1, 3, 1, 1)
#         return (input - mean) / std

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        x = input
        x = x - self.mean
        x = x / self.std
        return x


def ensemble_inference(net, x_in, num_classes=10):
    batch = x_in.size(0)
    prev = 0
    prob = torch.FloatTensor(batch, num_classes).zero_().cuda()
    answer = []
    with torch.no_grad():
        for n in [100]:
            for _ in range(n - prev):
                p = nn.Softmax(dim=1)(net(x_in)[0])
                prob.add_(p)
            answer.append(prob.clone())
            prev = n
        for i, a in enumerate(answer):
            answer[i] = torch.max(a, dim=1)[1]
    return answer


def process(rootpath, noise=0.3, attack='fgsm'):
    # normalize = Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # normalize.cuda()
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if 'CIFAR-100' in rootpath:
        valset = datasets.CIFAR100(root='/slstore/tianfeng/data', train=False, download=True, transform=transform_test)

        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, pin_memory=True,
                                                num_workers=0)
        num_classes = 100
    elif 'CIFAR-10' in rootpath:
        valset = datasets.CIFAR10(root='/slstore/tianfeng/data', train=False, download=True, transform=transform_test)

        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, pin_memory=True,
                                                num_workers=0)
        num_classes = 10
    elif 'FashionMNIST' in rootpath:

        test_kwargs = {'root': '/slstore/tianfeng/data', 'train': False, 'transform': transform_test, 'download': False}
        valset = datasets.FashionMNIST(**test_kwargs)
        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                                num_workers=0)
        num_classes = 10
    elif 'KMNIST' in rootpath:

        test_kwargs = {'root': '/slstore/tianfeng/data', 'train': False, 'transform': transform_test, 'download': False}
        valset = datasets.KMNIST(**test_kwargs)
        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                                num_workers=0)
        num_classes = 10
    elif 'MNIST' in rootpath:

        test_kwargs = {'root': '/slstore/tianfeng/data', 'train': False, 'transform': transform_test, 'download': False}
        valset = datasets.MNIST(**test_kwargs)
        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                                num_workers=0)
        num_classes = 10
    else:
        raise NotImplementedError

    if 'resnet20' in rootpath:
        net = ResNetWrapper(N=50000, num_classes=num_classes, net='resnet20')
        net.load(rootpath + "resnet20.pt")
    elif 'vgg' in rootpath:
        net = VGG16VIWrapper(N=50000, num_classes=num_classes)
        net.load(rootpath + "vgg.pt")
    elif 'lenet' in rootpath:
        net = LeNetVIWrapper(N=60000, num_classes=num_classes)
        net.load(rootpath + "lenet.pt")
    else:
        raise NotImplementedError

    if "MNIST" in rootpath:
        model = nn.Sequential(Normalize((0.5,), (0.5,)).cuda(), net.model)
    else:
        model = net.model
    model.eval()

    if attack == "pgd":
        attack = PGD(model, eps=noise)
    elif attack == "fgsm":
        attack = FGSM(model, eps=noise)
    else:
        raise NotImplementedError
    correct = 0
    total = 0
    batch = 0
    for it, (x, y) in tqdm.tqdm(enumerate(valloader)):
        x, y = x.cuda(), y.cuda()
        x_adv = attack(x, y)
        pred = ensemble_inference(model, x_adv, num_classes=num_classes)
        for i, p in enumerate(pred):
            correct += torch.sum(p.eq(y)).item()
        total += y.numel()
        batch += 1

    return correct


if __name__ == '__main__':
    rootrootpath = "/slstore/tianfeng/IB_BNN/figures/"
    rootpaths = [
        "lenet_seed42_l0.01_epoch200_datasetKMNIST_modellenet_samplings1_ratio0.0_inits-1.8971199848858813/",
        "lenet_seed43_l0.01_epoch200_datasetKMNIST_modellenet_samplings1_ratio0.0_inits-1.8971199848858813/",
        "lenet_seed44_l0.01_epoch200_datasetKMNIST_modellenet_samplings1_ratio0.0_inits-1.8971199848858813/",
        "lenet_seed42_l0.01_epoch200_datasetKMNIST_modellenet_samplings1_ratio0.0001_inits-1.8971199848858813/",
        "lenet_seed43_l0.01_epoch200_datasetKMNIST_modellenet_samplings1_ratio0.0001_inits-1.8971199848858813/",
        "lenet_seed44_l0.01_epoch200_datasetKMNIST_modellenet_samplings1_ratio0.0001_inits-1.8971199848858813/",
    ]
    noises = list(reversed([0.3, 0.1, 8 / 255, 4 / 255, 2 / 255, 1 / 255, 0.0]))
    for rootpath in rootpaths:
        res = []
        for noise in noises:
            r = process(rootrootpath + rootpath, noise, attack=args.attack)
            res.append(r)
            print(rootpath, noise)
            print(r)
        print(res)
