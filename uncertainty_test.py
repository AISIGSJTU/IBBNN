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
    else:
        raise NotImplementedError

    Has, Hes = np.array([]), np.array([])
    preds, labels = np.array([]), np.array([])
    for it, (x, y) in enumerate(valloader):
        x = x.cuda()
        pred, Ha, He = net.sample_predict(x, Nsamples=10)
        prob = nn.Softmax(dim=2)(pred)
        prob_sum = torch.sum(prob, dim=0)
        assert prob_sum.shape == torch.Size([100, num_classes])
        pred = torch.max(prob_sum, dim=1)[1]

        Has = np.concatenate([Has, Ha.cpu().detach()])
        Hes = np.concatenate([Hes, He.cpu().detach()])
        preds = np.concatenate([preds, pred.cpu().detach()])
        labels = np.concatenate([labels, y.detach()])
    return preds, labels, Has, Hes


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='uncertainty test')
    # args = parser.parse_args()
    rootpaths = [
        "/slstore/tianfeng/IB_BNN/figures/CIFAR-100_vgg/test_seed42_l0.01_epoch200_datasetCIFAR-100_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/",
        "/slstore/tianfeng/IB_BNN/figures/CIFAR-100_vgg/new_seed42_l0.01_epoch200_datasetCIFAR-100_modelvgg_samplings1_ratio0.01_inits-1.8971199848858813/"
    ]
    predsb, labelsb, Hasb, Hesb = process(rootpaths[0])
    preds, labels, Has, Hes = process(rootpaths[1])
    fontsize = 10
    legendsize = 14
    plt.clf()
    plt.tight_layout()
    plt.hist(Hes, alpha=0.7)
    plt.hist(Hesb, alpha=0.7)
    plt.legend(['with VIB', 'w/o. VIB'], frameon=False, prop={'size': legendsize})
    plt.xticks(fontsize=fontsize)
    plt.xlabel('Uncertainty', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Frequency', fontsize=fontsize)
    plt.title("Epistemic Uncertainty")
    plt.show()

    # for rootpath in rootpaths:
    # #     predsb, labelsb, Hasb, Hesb = process(rootpath, preprocess=transforms.functional(kernel_size=[3, 3]))
    #     preds, labels, Has, Hes = process(rootpath)
    #
    #     fontsize = 10
    #     legendsize = 14
    #     plt.clf()
    #     plt.tight_layout()
    #     plt.hist(Hes, alpha=0.7)
    #     plt.hist(Hesb, alpha=0.7)
    #     plt.legend(['Normal images', 'Blurred images'], frameon=False, prop={'size': legendsize})
    #     plt.xticks(fontsize=fontsize)
    #     plt.xlabel('Uncertainty', fontsize=fontsize)
    #     plt.yticks(fontsize=fontsize)
    #     plt.ylabel('Frequency', fontsize=fontsize)
    #     plt.title("Epistemic Uncertainty")
    #     plt.savefig(rootpath + "Hes.pdf")
    #
    #     fontsize = 10
    #     legendsize = 14
    #     plt.clf()
    #     plt.tight_layout()
    #     plt.hist(Has, alpha=0.7)
    #     plt.hist(Hasb, alpha=0.7)
    #     plt.legend(['Normal images', 'Blurred images'], frameon=False, prop={'size': legendsize})
    #     plt.xticks(fontsize=fontsize)
    #     plt.xlabel('Uncertainty', fontsize=fontsize)
    #     plt.yticks(fontsize=fontsize)
    #     plt.ylabel('Frequency', fontsize=fontsize)
    #     plt.title("Aleatoric Uncertainty")
    #     plt.savefig(rootpath + "Has.pdf")

        # mix_pred = [(i, j, Ha + He) for i, j, Ha, He in zip(preds, labels, Has, Hes)]
        # mix_pred.sort(key=lambda x: x[2])
        # corrects = [i[0] == i[1] for i in mix_pred]
        #
        # step_size = 1000
        # s = [sum(corrects[i:i + 1000]) for i in range(0, 10000, 1000)]
        # print(s)
        # accus = s[0:1]
        # for i in s[1:]:
        #     accus.append(accus[-1] + i)
        # accus = [x / (i * 1000 + 1000) for i, x in enumerate(accus)]
        # with open(rootpath + "retained_acc.pkl", "wb") as file:
        #     pickle.dump(accus, file)
        #
        # plt.style.use('ggplot')
        # fontsize = 10
        # legendsize = 14
        # plt.clf()
        # plt.tight_layout()
        # plt.plot(range(1000, 11000, 1000), accus, marker="*")
        # plt.title("Accuracy with retained data")
        # plt.xticks(fontsize=fontsize)
        # plt.xlabel("Predictions retained")
        # plt.yticks(fontsize=fontsize)
        # plt.ylabel("Accuracy")
        # plt.savefig(rootpath + "retained_acc.png")
        #
        # with open(rootpath + "retained_acc.txt", "w") as file:
        #     for accu in accus:
        #         file.write("%.4f " % accu)