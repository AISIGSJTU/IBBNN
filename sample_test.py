import pickle

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model import *


def ensemble_inference(net, x_in, num_classes=10, warming=False):
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
            if warming:
                break
        for i, a in enumerate(answer):
            answer[i] = torch.max(a, dim=1)[1]
    return answer


def process(rootpath, warming=False):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

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
    elif 'FashionMNIST' in rootpath:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)), ])

        test_kwargs = {'root': '/slstore/tianfeng/data', 'train': False, 'transform': transform, 'download': False}
        valset = datasets.FashionMNIST(**test_kwargs)
        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                                num_workers=0)
        num_classes = 10
    elif 'KMNIST' in rootpath:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)), ])

        test_kwargs = {'root': '/slstore/tianfeng/data', 'train': False, 'transform': transform, 'download': False}
        valset = datasets.KMNIST(**test_kwargs)
        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                                num_workers=0)
        num_classes = 10
    elif 'MNIST' in rootpath:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)), ])

        test_kwargs = {'root': '/slstore/tianfeng/data', 'train': False, 'transform': transform, 'download': False}
        valset = datasets.MNIST(**test_kwargs)
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
    elif 'toynet' in rootpath:
        net = ToyNetWrapper(N=60000, vi="bnnvib" in rootpath)
    elif 'lenet' in rootpath:
        net = LeNetVIWrapper(N=60000)
        net.load(rootpath + 'lenet.pt')
    else:
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss().cuda()

    precs = []
    for i in range(100):
        logging.info(f"========= {i} ==========")
        print(f"========= {i} ==========")
        with torch.no_grad():
            for module in net.model.modules():
                if type(module).__name__ in ["RandConv2d", "RandLinear", "RandBatchNorm2d"]:
                    module.eps_weight.normal_()
                    if module.eps_bias is not None:
                        module.eps_bias.normal_()
        _, prec = validate(valloader, net.model, criterion, fix=True, half=False)
        precs.append(prec)
        if warming:
            break

    net.model.eval()
    correct = 0
    total = 0
    batch = 0
    for it, (x, y) in enumerate(valloader):
        x, y = x.cuda(), y.cuda()
        pred = ensemble_inference(net.model, x, num_classes=num_classes, warming=warming)
        for i, p in enumerate(pred):
            correct += torch.sum(p.eq(y)).item()
        logging.info(f"{correct}, {it}")
        print(f"{correct}, {it + 1}, {correct / (it + 1)}")
        total += y.numel()
        batch += 1
        if warming:
            break

    logging.info(f"{correct}")
    print(f"{correct}")

    with open(rootpath + "metrics/precs100.pkl", "wb") as file:
        pickle.dump(precs, file)

    plt.clf()
    plt.style.use('ggplot')
    plt.hist(precs)
    plt.xlabel(f"precs ({min(precs)} - {max(precs)}) em {correct}")
    plt.ylabel("frequency")
    plt.savefig(rootpath + "figures/precs100.png")
    plt.savefig(rootpath + f"figures/precs{correct}.png")
    plt.clf()


if __name__ == '__main__':
    rootrootpath = ""
    rootpaths = [
        # "CIFAR-100-resnet56/normal_0.02alpha_5iters/ratio_seed42_l0.01_epoch200_optadam_ratio0.8_datasetCIFAR-100_modelresnet56_samplings1_sebr0.0/",
        # "/slstore/tianfeng/IB_BNN/figures/lenet_seed42_l0.01_epoch200_datasetFashionMNIST_modellenet_samplings1_ratio0.0_inits-1.8971199848858813/",
        "/slstore/tianfeng/IB_BNN/figures/FashionMNIST_lenet/baselines/lenet_seed43_l0.01_epoch200_datasetFashionMNIST_modellenet_samplings1_ratio0.0_inits-1.8971199848858813/",
        "/slstore/tianfeng/IB_BNN/figures/FashionMNIST_lenet/baselines/lenet_seed44_l0.01_epoch200_datasetFashionMNIST_modellenet_samplings1_ratio0.0_inits-1.8971199848858813/",
    ]
    for rootpath in rootpaths:
        process(rootrootpath + rootpath)
