import os
import random
import functools
import shutil
import datetime
import logging
import math

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
import pickle

from utils import *
from model import *
from plot import plot
from sample_test import process

parser = argparse.ArgumentParser(description='variance scaled weight perturbation')
parser.add_argument("-s", "--seed", type=int, default=42, help="The random seed")
parser.add_argument("-l", "--learning_rate", type=float, default=0.01, help="The learning rate")
parser.add_argument("-e", "--epoch", type=int, default=200, help="The total epoch")
parser.add_argument('--ratio', default=0.0, type=float, help='hyperparameter lambda for IB')
parser.add_argument('--dataset', default='CIFAR-10', type=str, help='dataset')
parser.add_argument('--model', default='resnet20', type=str, help='network structure')
parser.add_argument('--samplings', default=1, type=int, help='sampling times')
parser.add_argument('--pretrain', default=False, type=bool, help='use pre training and Bayes finetune')
parser.add_argument('--notvi', help='not use BNN', dest="notvi", action='store_true')
parser.add_argument('--sigma0', default=0.15, type=float, help='sigma prior of hyperparameters')
parser.add_argument('--inits', default=0.15, type=float, help='init of hyperparameters')
parser.add_argument('--warming', help='warming', dest="warming", action='store_true')
parser.add_argument('--mode', default='none', type=str, help='mode for vib')
parser.add_argument('--after_epoch', default=-1, type=int, help='after_epoch')
parser.add_argument('--before_epoch', default=500, type=int, help='before_epoch')
parser.add_argument('--lambda2', default=0.0, type=float, help='penalty on variance')
parser.add_argument('--asratio', default=0.0, type=float, help='hyperparameter lambda for AS')
args = parser.parse_args()
args.inits = math.log(args.inits)
cudnn.benchmark = True
datapath = "/slstore/tianfeng/data"
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print("Start time:", start_time.isoformat())
    if args.warming:
        args.epoch = 1
        print("warming")
        signature = "warming"
    else:
        signature = "vib"
    seed = args.seed
    print("Use random seed ", seed)
    print("Use learning rate ", args.learning_rate)
    print("Use epoch", args.epoch)
    print("Use dataset", args.dataset)
    print("Sampling times", args.samplings)
    print("sigma0", args.sigma0)
    print("inits", args.inits)
    print("ratio", args.ratio)
    rootpath = f"results/{signature}"
    rootpath += f"_seed{seed}"
    rootpath += f'_l{args.learning_rate}'
    rootpath += f'_epoch{args.epoch}'
    rootpath += f'_dataset{args.dataset}'
    rootpath += f'_model{args.model}'
    rootpath += f'_samplings{args.samplings}'
    rootpath += f"_ratio{args.ratio}" if args.ratio != 0.0 else ""
    rootpath += f"_asratio{args.asratio}" if args.asratio != 0.0 else ""
    rootpath += f"_lambda2_{args.lambda2}" if args.lambda2 != 0.0 else ""
    rootpath += f'_sigma0{args.sigma0}' if args.sigma0 != 0.15 else ""
    rootpath += f'_inits{args.inits}' if args.inits != 0.15 else ""
    rootpath += f'_after_epoch{args.after_epoch}' if args.after_epoch != 0 else ""
    rootpath += f'_before_epoch{args.before_epoch}' if args.before_epoch != 500 else ""
    rootpath += "_pretrain" if args.pretrain else ""
    rootpath += "_notvi" if args.notvi else ""
    rootpath += f"_mode{args.mode}" if args.mode != "none" else ""
    rootpath += '/'
    if not os.path.isdir(rootpath):
        print("rootpath: ", rootpath)
        os.mkdir(rootpath)
        os.mkdir(rootpath + "codes")
        os.mkdir(rootpath + "metrics")
        os.mkdir(rootpath + "figures")
        os.mkdir(rootpath + "debug")
        os.mkdir(rootpath + "metrics/IBss")
        shutil.copyfile("main.py", rootpath + "codes/main_bak.py")
        shutil.copyfile("utils.py", rootpath + "codes/utils_bak.py")
        shutil.copytree("models", rootpath + "codes/models_bak")
    else:
        print("rootpath: ", rootpath)
        print("WARNING: rootpath exists")

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=rootpath + 'log.log', level=logging.INFO, format=LOG_FORMAT)

    logging.info(rootpath)
    logging.info("rootpath: " + rootpath)

    set_seed(seed)

    log_interval = 1

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # normalize,
    ])

    if args.dataset == 'CIFAR-10':
        trainset = datasets.CIFAR10(root=datapath, train=True, download=True, transform=transform_train)
        valset = datasets.CIFAR10(root=datapath, train=False, download=True, transform=transform_test)
        num_classes = 10
    elif args.dataset == 'CIFAR-100':
        trainset = datasets.CIFAR100(root=datapath, train=True, download=True,
                                     transform=transform_train)
        valset = datasets.CIFAR100(root=datapath, train=False, download=True, transform=transform_test)
        num_classes = 100
    elif args.dataset == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)), ])
        train_kwargs = {'root': datapath, 'train': True, 'transform': transform, 'download': True}
        test_kwargs = {'root': datapath, 'train': False, 'transform': transform, 'download': False}
        trainset = datasets.FashionMNIST(**train_kwargs)
        valset = datasets.FashionMNIST(**test_kwargs)
        num_classes = 10
    elif args.dataset == 'KMNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)), ])
        train_kwargs = {'root': datapath, 'train': True, 'transform': transform, 'download': True}
        test_kwargs = {'root': datapath, 'train': False, 'transform': transform, 'download': False}
        trainset = datasets.KMNIST(**train_kwargs)
        valset = datasets.KMNIST(**test_kwargs)
        num_classes = 10
    elif args.dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)), ])
        train_kwargs = {'root': datapath, 'train': True, 'transform': transform, 'download': True}
        test_kwargs = {'root': datapath, 'train': False, 'transform': transform, 'download': False}
        trainset = datasets.MNIST(**train_kwargs)
        valset = datasets.MNIST(**test_kwargs)
        num_classes = 10
    elif args.dataset == "advCIFAR-10":
        class CIFAR_load(torch.utils.data.Dataset):
            def __init__(self, root, baseset, dummy_root='~/data', split='train', download=False, **kwargs):
                self.baseset = baseset
                self.transform = self.baseset.transform
                self.samples = os.listdir(os.path.join(root, 'data'))
                self.root = root

            def __len__(self):
                return len(self.baseset)

            def __getitem__(self, idx):
                true_index = int(self.samples[idx].split('.')[0])
                true_img, label = self.baseset[true_index]
                return self.transform(Image.open(os.path.join(self.root, 'data',
                                                              self.samples[idx]))), label #, true_img


        baseset = datasets.CIFAR10(
            root=datapath, train=True, download=False, transform=transform_train)
        trainset = CIFAR_load(root="/slstore/tianfeng/adversarial_poison/figures/dataset", baseset=baseset)

        valset = datasets.CIFAR10(
            root=datapath, train=False, download=False, transform=transform_test)
        num_classes = 10
    else:
        raise NotImplementedError('Invalid dataset')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,
                                              num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False,
                                            num_workers=0)

    if 'resnet' in args.model:
        net = ResNetWrapper(N=len(trainset), num_classes=num_classes, net=args.model, vi=not args.notvi,
                            sigma0=args.sigma0, inits=args.inits)
    elif 'vgg' in args.model:
        net = VGG16VIWrapper(N=len(trainset), num_classes=num_classes, vi=not args.notvi, sigma0=args.sigma0,
                             inits=args.inits)
    elif 'toynet' in args.model:
        net = ToyNetWrapper(N=len(trainset), vi=not args.notvi, mode=args.mode)
    elif 'lenet' in args.model:
        net = LeNetVIWrapper(N=len(trainset), vi=not args.notvi, sigma0=args.sigma0, inits=args.inits,
                             num_classes=num_classes)
    else:
        raise NotImplementedError('Invalid model')

    train_losses = []
    train_precs = []
    test_losses = []
    test_precs = []

    lr = args.learning_rate

    if args.pretrain:
        pretrain_path = f"pretrain/{args.dataset}-{args.model}/{args.model}.pt"
        net_dict = net.model.state_dict()
        pretrain_dict = torch.load(pretrain_path, map_location='cuda:0')[
            'state_dict'].items()
        net_dict.update(
            {k.replace('weight', 'mu_weight').replace('bias', 'mu_bias'): v for
             k, v in pretrain_dict})
        net.model.load_state_dict(net_dict)
        print("Load from " + pretrain_path)
        net.validate(valloader, sample=False)

    optimizer = torch.optim.Adam(net.model.parameters(), lr=lr)
    for epoch in range(args.epoch):
        if epoch in [80, 140, 180]:
            net.save(rootpath + f"{args.model}_{epoch}epoch.pt")
            lr /= 10
            optimizer = torch.optim.Adam(net.model.parameters(), lr=lr)

        try:
            if epoch > args.after_epoch and epoch < args.before_epoch:
                if 'toynet' in args.model:
                    loss, prec = net.fit(trainloader, epoch=epoch, optimizer=optimizer, samplings=args.samplings,
                                         asratio=args.asratio)
                else:
                    loss, prec, IBss = net.fit(trainloader, epoch=epoch, optimizer=optimizer, samplings=args.samplings,
                                         ratio=args.ratio, lambda2=args.lambda2)
            else:
                loss, prec, IBss = net.fit(trainloader, epoch=epoch, optimizer=optimizer, samplings=args.samplings,
                                           ratio=0, lambda2=args.lambda2)
        except RuntimeError as inst:
            kl_out, sig_out, mu_out = inst.args
            torch.save(kl_out, rootpath + "debug/kl_out.pt")
            torch.save(sig_out, rootpath + "debug/sig_out.pt")
            torch.save(mu_out, rootpath + "debug/mu_out.pt")
            if args.ratio > 1e-10:
                raise
        if not 'toynet' in args.model:
            torch.save(IBss, rootpath + f'metrics/IBss/{epoch}.pt')
        train_precs.append(prec)
        train_losses.append(loss)

        loss, prec = net.validate(valloader)
        test_precs.append(prec)
        test_losses.append(loss)
    print(f"Best prec,  {max(test_precs)}")
    logging.info(f"Best prec,  {max(test_precs)}")
    net.save(rootpath + f"{args.model}.pt")

    training_metrics = ["train_precs", "train_losses", "test_precs", "test_losses"]
    for m in training_metrics:
        with open(rootpath + "metrics/" + f"{m}.pkl", 'wb') as file:
            pickle.dump(eval(f"{m}"), file)

    for m in training_metrics:
        plot(rootpath + "metrics/" + f"{m}.pkl", rootpath + "figures/" + m + ".png")
        plot(rootpath + "metrics/" + f"{m}.pkl", rootpath + "figures/" + m + ".pdf")

    if not args.notvi:
        process(rootpath, warming=args.warming)

    end_time = datetime.datetime.now()
    logging.info(f"End time: {end_time.isoformat()}")
    logging.info(f"Total time: {end_time - start_time}")
