import pickle
import argparse
from functools import reduce

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from model import *


def process(rootpath, preprocess=None):
    print(rootpath)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    if preprocess:
        transform_test = transforms.Compose([
            preprocess,
            transforms.ToTensor(),
            # normalize,
        ])
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ])

    if "CIFAR-100" in rootpath:
        valset = datasets.CIFAR100(root='/slstore/tianfeng/data', train=False, download=True, transform=transform_test)

        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, pin_memory=True,
                                                num_workers=0)
        num_classes = 100
    elif "CIFAR-10" in rootpath:
        valset = datasets.CIFAR10(root='/slstore/tianfeng/data', train=False, download=True, transform=transform_test)

        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, pin_memory=True,
                                                num_workers=0)
        num_classes = 10
    elif "FashionMNIST" in rootpath:
        datapath = '/slstore/tianfeng/data'
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)), ])
        test_kwargs = {'root': datapath, 'train': False, 'transform': transform, 'download': False}
        valset = datasets.FashionMNIST(**test_kwargs)
        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                            num_workers=0)
        num_classes = 10
    else:
        raise NotImplementedError

    if 'resnet20' in rootpath:
        net = ResNetWrapper(N=50000, num_classes=num_classes, net='resnet20')
        net.load(rootpath + "resnet20.pt")
    elif 'resnet56' in rootpath:
        net = ResNetWrapper(N=50000, num_classes=num_classes, net='resnet56')
        net.load(rootpath + "resnet56.pt")
    elif 'vgg' in rootpath:
        net = VGG16VIWrapper(N=50000, num_classes=num_classes)
        net.load(rootpath + "vgg.pt")
    elif 'lenet' in rootpath:
        net = LeNetVIWrapper(N=60000, num_classes=num_classes)
        net.load(rootpath + "lenet.pt")
    else:
        raise NotImplementedError

    Has, Hes = np.array([]), np.array([])
    preds, labels = np.array([]), np.array([])
    ibs = np.array([])
    for it, (x, y) in enumerate(valloader):
        x = x.cuda()
        pred, Ha, He = net.sample_predict(x, Nsamples=10)
        prob = nn.Softmax(dim=2)(pred)
        prob_sum = torch.sum(prob, dim=0)
        assert prob_sum.shape == torch.Size([100, num_classes])
        pred = torch.max(prob_sum, dim=1)[1]

        if "vgg" in rootpath:
            ib = net.model.features[20].kl_output(mean_batch=False)
        if "lenet" in rootpath:
            ib = net.model.conv_layers[4].kl_output(mean_batch=False)

        Has = np.concatenate([Has, Ha.cpu().detach()])
        Hes = np.concatenate([Hes, He.cpu().detach()])
        preds = np.concatenate([preds, pred.cpu().detach()])
        labels = np.concatenate([labels, y.detach()])
        ibs = np.concatenate([ibs, ib.cpu().detach()])
    return preds, labels, Has, Hes, ibs


# rootpaths = [
#     "/slstore/tianfeng/IB_BNN/figures/CIFAR-100_vgg/baseline/test_seed42_l0.01_epoch200_datasetCIFAR-100_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/",
#     "/slstore/tianfeng/IB_BNN/figures/CIFAR-100_vgg/lambda2/variance_seed42_l0.01_epoch200_datasetCIFAR-100_modelvgg_samplings1_ratio0.0_lambda2_0.0001_inits-1.8971199848858813/",
#     ]
# preds0, labels0, Has0, Hes0, ib0 = process(rootpaths[0])
# preds1, labels1, Has1, Hes1, ib1 = process(rootpaths[1])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     # normalize,
# ])
#
# valset = datasets.CIFAR100(root='/slstore/tianfeng/data', train=False, download=True, transform=transform_test)
#
# valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, pin_memory=True,
#                                         num_workers=0)
# num_classes = 100
#
# idxs = np.argpartition(ib0, 5)[:5]
# for idx in idxs:
#     data = valset[idx]
#     plt.imshow(data[0].permute([1, 2, 0]))
#     plt.show()
#     print(valset.classes[data[1]])
#     print(valset.classes[int(preds0[idx])], valset.classes[int(preds1[idx])])
#     print(ib0[idx], ib1[idx])



# rootpaths = [
#     "/slstore/tianfeng/IB_BNN/figures/CIFAR-10_vgg/baseline/new_seed42_l0.01_epoch200_datasetCIFAR-10_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/",
#     "/slstore/tianfeng/IB_BNN/figures/CIFAR-10_vgg/lambda2/variance_seed42_l0.01_epoch200_datasetCIFAR-10_modelvgg_samplings1_ratio0.0_lambda2_0.0001_inits-1.8971199848858813/",
#     ]
# preds0, labels0, Has0, Hes0, ib0 = process(rootpaths[0])
# preds1, labels1, Has1, Hes1, ib1 = process(rootpaths[1])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     # normalize,
# ])
#
# valset = datasets.CIFAR10(root='/slstore/tianfeng/data', train=False, download=True, transform=transform_test)
#
# valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, pin_memory=True,
#                                         num_workers=0)
# num_classes = 10
#
# idxs = np.argpartition(ib0, 10)[:10]
# for idx in idxs:
#     data = valset[idx]
#     plt.imshow(data[0].permute([1, 2, 0]))
#     plt.show()
#     print(valset.classes[data[1]])
#     print(valset.classes[int(preds0[idx])], valset.classes[int(preds1[idx])])
#     print(ib0[idx], ib1[idx])


rootpaths = [
    "/slstore/tianfeng/IB_BNN/figures/FashionMNIST_lenet/baselines/lenet_seed42_l0.01_epoch200_datasetFashionMNIST_modellenet_samplings1_ratio0.0_inits-1.8971199848858813/",
    "/slstore/tianfeng/IB_BNN/figures/FashionMNIST_lenet/lambda2/lenet_seed42_l0.01_epoch200_datasetFashionMNIST_modellenet_samplings1_ratio0.0_lambda2_0.0001_inits-1.8971199848858813/",
    ]
preds0, labels0, Has0, Hes0, ib0 = process(rootpaths[0])
preds1, labels1, Has1, Hes1, ib1 = process(rootpaths[1])


datapath = '/slstore/tianfeng/data'
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)), ])
test_kwargs = {'root': datapath, 'train': False, 'transform': transform, 'download': False}
valset = datasets.FashionMNIST(**test_kwargs)
valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                        num_workers=0)
num_classes = 10

idxs = np.argpartition(ib0, 5)[:5]
for idx in idxs:
    data = valset[idx]
    plt.imshow(data[0].permute([1, 2, 0]))
    plt.show()
    print(valset.classes[data[1]])
    print(valset.classes[int(preds0[idx])], valset.classes[int(preds1[idx])])
    print(ib0[idx], ib1[idx])