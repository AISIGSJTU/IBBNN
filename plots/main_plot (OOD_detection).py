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

class RandomNoiseDataset(VisionDataset):
    def __init__(self, root, shape, transform=None, target_transform=None):
        super(RandomNoiseDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.data = torch.rand(shape)

    def __getitem__(self, index):
        img = self.data[index]
        return img, 0

    def __len__(self) -> int:
        return len(self.data)


def process(dataloader, net):
    Has, Hes = np.array([]), np.array([])
    preds, labels = np.array([]), np.array([])
    ibs = np.array([])
    for it, (x, y) in enumerate(dataloader):
        x = x.cuda()
        pred, Ha, He = net.sample_predict(x, Nsamples=10)
        prob = nn.Softmax(dim=2)(pred)
        prob_sum = torch.sum(prob, dim=0)
        # assert prob_sum.shape == torch.Size([100, num_classes])
        pred = torch.max(prob_sum, dim=1)[1]
        # ib = net.model.conv_layers[4].kl_output(mean_batch=False)
        ib = net.model.features[20].kl_output(mean_batch=False)
        Has = np.concatenate([Has, Ha.cpu().detach()])
        Hes = np.concatenate([Hes, He.cpu().detach()])
        preds = np.concatenate([preds, pred.cpu().detach()])
        labels = np.concatenate([labels, y.detach()])
        ibs = np.concatenate([ibs, ib.cpu().detach()])
    return preds, labels, Has, Hes, ibs


rootpaths = [
    "/slstore/tianfeng/IB_BNN/figures/CIFAR-10_vgg/baseline/new_seed42_l0.01_epoch200_datasetCIFAR-10_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/",
    "/slstore/tianfeng/IB_BNN/figures/CIFAR-100_vgg/baseline/test_seed42_l0.01_epoch200_datasetCIFAR-100_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/"
    # "/slstore/tianfeng/IB_BNN/figures/FashionMNIST_lenet/baselines/lenet_seed42_l0.01_epoch200_datasetFashionMNIST_modellenet_samplings1_ratio0.0_inits-1.8971199848858813/",
    # "/slstore/tianfeng/IB_BNN/figures/KMNIST_lenet/baseline/lenet_seed42_l0.01_epoch200_datasetKMNIST_modellenet_samplings1_ratio0.0_inits-1.8971199848858813/",
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

# c10netrandomdata = process(dataloader=randloader, net=c10net)
# c100netrandomdata = process(dataloader=randloader, net=c100net)
c10netc10data = process(dataloader=c10valloader, net=c10net)
c100netc10data = process(dataloader=c10valloader, net=c100net)
c10netc100data = process(dataloader=c100valloader, net=c10net)
c100netc100data = process(dataloader=c100valloader, net=c100net)
c10netsvhndata = process(dataloader=SVHNvalloader, net=c10net)
c100netsvhndata = process(dataloader=SVHNvalloader, net=c100net)

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (6.0, 4.5)

plt.clf()
plt.hist(c10netc10data[4], alpha=0.8, density=True)
plt.hist(c10netc100data[4], alpha=0.8, density=True)
plt.hist(c10netsvhndata[4], alpha=0.8, density=True)
plt.legend(["CIFAR-10", "CIFAR-100", "SVHN"])
plt.xlabel("Information Bound")
plt.ylabel("Frequency")
plt.savefig("../figures/c10netood.pdf")
plt.show()

plt.clf()
plt.hist(c100netc10data[4], alpha=0.8, density=True, bins=15)
plt.hist(c100netc100data[4], alpha=0.8, density=True, bins=15)
plt.hist(c100netsvhndata[4], alpha=0.8, density=True)
plt.legend(["CIFAR-10", "CIFAR-100", "SVHN"])
plt.xlabel("Information Bound")
plt.ylabel("Frequency")
plt.savefig("../figures/c100netood.pdf")
plt.show()
