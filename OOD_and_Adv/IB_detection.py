import pickle
import argparse
from functools import reduce

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets.vision import VisionDataset
from PIL import Image

from model import *
from attacks.attack import PGD, FGSM


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


class RandomNoiseDataset(VisionDataset):
    def __init__(self, root, shape, transform=None, target_transform=None):
        super(RandomNoiseDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.data = torch.rand(shape)

    def __getitem__(self, index):
        img = self.data[index]
        return img, 0

    def __len__(self) -> int:
        return len(self.data)


def process(dataloader, net,  n_layers, attack=None, noise=0.0):
    Has, Hes = np.array([]), np.array([])
    preds, labels = np.array([]), np.array([])
    ibss = [np.array([]) for i in n_layers]

    if "LeNet" in str(type(net)):
        model = nn.Sequential(Normalize((0.5,), (0.5,)).cuda(), net.model)
    elif "VGG" in str(type(net)):
        model = net.model
    model.eval()

    if attack == "pgd":
        attack = PGD(model, eps=noise)
    elif attack == "fgsm":
        attack = FGSM(model, eps=noise)
    else:
        attack = None

    for it, (x, y) in enumerate(dataloader):
        xc = x.cuda()
        if attack:
            y = y.cuda()
            x_adv = attack(xc, y)
            pred, Ha, He = net.sample_predict(x_adv, Nsamples=10)
        else:
            pred, Ha, He = net.sample_predict(xc, Nsamples=10)
        prob = nn.Softmax(dim=2)(pred)
        prob_sum = torch.sum(prob, dim=0)
        # assert prob_sum.shape == torch.Size([100, num_classes])
        pred = torch.max(prob_sum, dim=1)[1]
        # ib = net.model.conv_layers[4].kl_output(mean_batch=False)
        for i, n_layer in enumerate(n_layers):
            ib = net.model.features[n_layer].kl_output(mean_batch=False)
            ibss[i] = np.concatenate([ibss[i], ib.cpu().detach()])
        Has = np.concatenate([Has, Ha.cpu().detach()])
        Hes = np.concatenate([Hes, He.cpu().detach()])
        preds = np.concatenate([preds, pred.cpu().detach()])
        labels = np.concatenate([labels, y.cpu().detach()])
    return preds, labels, Has, Hes, ibss


rootpaths = [
    "/slstore/tianfeng/IB_BNN/figures/CIFAR-10_vgg/baseline/new_seed42_l0.01_epoch200_datasetCIFAR-10_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/",
    "/slstore/tianfeng/IB_BNN/figures/CIFAR-100_vgg/baseline/test_seed42_l0.01_epoch200_datasetCIFAR-100_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/"
]

datapath = "/slstore/tianfeng/data"

c10net = VGG16VIWrapper(N=50000)
c10net.load(filename=rootpaths[0] + 'vgg.pt')
c100net = VGG16VIWrapper(N=50000, num_classes=100)
c100net.load(filename=rootpaths[1] + 'vgg.pt')

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
c10valset = datasets.CIFAR10(root=datapath, train=False, download=True, transform=transform_test)
c10valloader = torch.utils.data.DataLoader(c10valset, batch_size=100, shuffle=False,
                                           num_workers=0)

c100valset = datasets.CIFAR100(root=datapath, train=False, download=True, transform=transform_test)
c100valloader = torch.utils.data.DataLoader(c100valset, batch_size=100, shuffle=False,
                                            num_workers=0)

randset = RandomNoiseDataset(root=datapath, shape=[10000, 3, 32, 32])
randloader = torch.utils.data.DataLoader(randset, batch_size=100, shuffle=False,
                                         num_workers=0)

SVHNvalset = datasets.SVHN(root=datapath, split='test', download=True, transform=transform_test)
SVHNvalloader = torch.utils.data.DataLoader(SVHNvalset, batch_size=100, shuffle=False,
                                            num_workers=0)
#

n_layers = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
c10netc10advdata = process(dataloader=c10valloader, net=c10net, attack='pgd', noise=0.1, n_layers=n_layers)
c10netc10data = process(dataloader=c10valloader, net=c10net, n_layers=n_layers)
c10netc100data = process(dataloader=c100valloader, net=c10net, n_layers=n_layers)
c10netsvhndata = process(dataloader=SVHNvalloader, net=c10net, n_layers=n_layers)
c10netrandomdata = process(dataloader=randloader, net=c10net, n_layers=n_layers)

c100netc100advdata = process(dataloader=c10valloader, net=c100net, attack='pgd', noise=0.1, n_layers=n_layers)
c100netc100data = process(dataloader=c10valloader, net=c100net, n_layers=n_layers)
c100netc10data = process(dataloader=c100valloader, net=c100net, n_layers=n_layers)
c100netsvhndata = process(dataloader=SVHNvalloader, net=c100net, n_layers=n_layers)
c100netrandomdata = process(dataloader=randloader, net=c100net, n_layers=n_layers)

# plt.style.use('ggplot')
# plt.rcParams['figure.figsize'] = (6.0, 4.5)
#
# plt.clf()
# plt.hist(c10netc10data[4], alpha=0.8, density=True)
# plt.hist(c10netc100data[4], alpha=0.8, density=True)
# plt.hist(c10netsvhndata[4], alpha=0.8, density=True)
# plt.legend(["CIFAR-10", "CIFAR-100", "SVHN"])
# plt.xlabel("Information Bound")
# plt.ylabel("Frequency")
# plt.savefig("../figures/c10netood.pdf")
# plt.show()
#
# plt.clf()
# plt.hist(c100netc10data[4], alpha=0.8, density=True, bins=15)
# plt.hist(c100netc100data[4], alpha=0.8, density=True, bins=15)
# plt.hist(c100netsvhndata[4], alpha=0.8, density=True)
# plt.legend(["CIFAR-10", "CIFAR-100", "SVHN"])
# plt.xlabel("Information Bound")
# plt.ylabel("Frequency")
# plt.savefig("../figures/c100netood.pdf")
# plt.show()


# for i, n_layer in  enumerate(n_layers):
#     print(c10netc10data[-1][i].mean())
#     print(c10netc100data[-1][i].mean())
#     print(c10netc10advdata[-1][i].mean())
# plt.clf()
# plt.plot(n_layers, [i.mean() for i in c10netc10data[-1]], marker='*')
# plt.plot(n_layers, [i.mean() for i in c10netc100data[-1]], marker='*')
# plt.plot(n_layers, [i.mean() for i in c10netsvhndata[-1]], marker='*')
# plt.plot(n_layers, [i.mean() for i in c10netc10advdata[-1]], marker='*')
# plt.plot(n_layers, [i.mean() for i in c10netrandomdata[-1]], marker='*')
# plt.legend(["c10", "c100", "svhn", "c10adv", "random"])
# plt.xlabel("layer")
# plt.ylabel("Information Bound")
# plt.title("C10 net")
# plt.savefig('1.png')
#
# plt.clf()
# plt.plot(n_layers, [i.mean() for i in c100netc100data[-1]], marker='*')
# plt.plot(n_layers, [i.mean() for i in c100netc10data[-1]], marker='*')
# plt.plot(n_layers, [i.mean() for i in c100netsvhndata[-1]], marker='*')
# plt.plot(n_layers, [i.mean() for i in c100netc100advdata[-1]], marker='*')
# plt.plot(n_layers, [i.mean() for i in c100netrandomdata[-1]], marker='*')
# plt.legend(["c100", "c10", "svhn", "c100adv", "random"])
# plt.xlabel("layer")
# plt.ylabel("Information Bound")
# plt.title("C100 net")
# plt.savefig('2.png')

# plt.clf()
# plt.hist(n_layers, [i.mean() for i in c10netc10data[-1]], marker='*')
# plt.hist(n_layers, [i.mean() for i in c10netc100data[-1]], marker='*')
# plt.hist(n_layers, [i.mean() for i in c10netsvhndata[-1]], marker='*')
# plt.hist(n_layers, [i.mean() for i in c10netc10advdata[-1]], marker='*')
# plt.hist(n_layers, [i.mean() for i in c10netrandomdata[-1]], marker='*')
# plt.legend(["c10", "c100", "svhn", "c10adv", "random"])
# plt.xlabel("layer")
# plt.ylabel("Information Bound")
# plt.title("C10 net")
# plt.savefig('1.png')


for i, n_layer in enumerate(n_layers):
    plt.clf()
    plt.hist(c10netc10data[-1][i], alpha=0.6)
    plt.hist(c10netc100data[-1][i], alpha=0.6)
    plt.hist(c10netsvhndata[-1][i], alpha=0.6)
    plt.hist(c10netc10advdata[-1][i], alpha=0.6)
    plt.hist(c10netrandomdata[-1][i], alpha=0.6)
    plt.legend(["c10", "c100", "svhn", "c10adv", "random"])
    plt.xlabel("Information Bound of c10 net")
    plt.ylabel("Frequency")
    plt.title(f"Layer = {n_layer}")
    plt.savefig(f"c10net-{n_layer}layer.png")

for i, n_layer in enumerate(n_layers):
    plt.clf()
    plt.hist(c100netc100data[-1][i], alpha=0.6)
    plt.hist(c100netc10data[-1][i], alpha=0.6)
    plt.hist(c100netsvhndata[-1][i], alpha=0.6)
    plt.hist(c100netc100advdata[-1][i], alpha=0.6)
    plt.hist(c100netrandomdata[-1][i], alpha=0.6)
    plt.legend(["c10", "c100", "svhn", "c10adv", "random"])
    plt.xlabel("Information Bound of c100 net")
    plt.ylabel("Frequency")
    plt.title(f"Layer = {n_layer}")
    plt.savefig(f"c100net-{n_layer}layer.png")