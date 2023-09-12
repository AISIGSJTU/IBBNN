import logging
import math

import numpy as np
import torch
import torch.nn as nn
import time
from torch.distributions import Categorical

from utils import *
from models.resnet_vi import resnet20_vi, resnet56_vi
from models.vgg_vi import VGG_vi
from models.vgg import VGG
from models.resnet import resnet20, resnet56
from models.toynet import ToyNet
from models.lenet_vi import Lenet_vi


class NetWrapper():
    def __init__(self):
        cprint('c', '\nNet:')
        self.model = None

    def fit(self, train_loader, optimizer):
        raise NotImplementedError

    def predict(self, test_loader):
        raise NotImplementedError

    def validate(self, val_loader):
        raise NotImplementedError

    def save(self, filename='checkpoint.pt'):
        state = {
            'state_dict': self.model.state_dict(),
        }
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        self.model.load_state_dict(state['state_dict'])


class ResNetWrapper(NetWrapper):
    def __init__(self, N, half=False, cuda=True, double=False, vi=True, num_classes=10, net='resnet20', sigma0=0.15,
                 inits=math.log(0.15)):
        super(ResNetWrapper).__init__()
        self.N = N
        self.vi = vi
        self.num_classes = num_classes
        if vi:
            if net == 'resnet20':
                self.model = resnet20_vi(N=N, num_classes=num_classes, sigma_0=sigma0, init_s=inits)
            elif net == 'resnet56':
                self.model = resnet56_vi(N=N, num_classes=num_classes, sigma_0=sigma0, init_s=inits)
        else:
            if net == 'resnet20':
                self.model = resnet20(num_classes=num_classes)
            elif net == 'resnet56':
                self.model = resnet56(num_classes=num_classes)
        self.half = half
        self.double = double
        if self.half:
            self.model.half()
        if self.double:
            self.model.double()
        if cuda:
            self.model.cuda()

    def fit(self, train_loader, lr=0.01, weight_decay=0, epoch=None, adv=None, optimizer='adam',
            samplings=1, ratio=0.0):
        if type(optimizer) is str:
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=0.9, weight_decay=weight_decay)
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
            else:
                raise ValueError("Optimizer {} not valid.".format(optimizer))
        logging.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss, prec, IBss = train(train_loader, self.model, optimizer, epoch, self.N, half=self.half, double=self.double,
                           vi=self.vi, samplings=samplings, ratio=ratio)

        return loss, prec, IBss

    def validate(self, val_loader, sample=True):
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()
        loss, prec = validate(val_loader, self.model, criterion, half=self.half, double=self.double, vi=self.vi,
                              sample=sample)
        return loss, prec

    def sample_predict(self, x, Nsamples):
        self.model.eval()
        with torch.no_grad():
            predictions = x.data.new(Nsamples, x.shape[0], self.num_classes)

            Hs = []
            for i in range(Nsamples):
                y, kl = self.model(x)
                predictions[i] = y

                output = nn.functional.softmax(y)
                H = torch.distributions.Categorical(probs=output).entropy()
                Hs.append(H)

            Ha = sum(Hs) / Nsamples
            He = sum(torch.abs(Ha - i) for i in Hs) / Nsamples
        return predictions, Ha, He


class VGG16VIWrapper(NetWrapper):
    def __init__(self, N, half=False, cuda=True, double=False, num_classes=10, vi=True, sigma0=0.15,
                 inits=math.log(0.15)):
        super(VGG16VIWrapper).__init__()
        self.N = N
        self.vi = vi
        self.num_classes = num_classes
        if not vi:
            self.model = VGG(nclass=num_classes)
        else:
            self.model = VGG_vi(sigma_0=sigma0, N=N, init_s=inits, nclass=num_classes)
        self.half = half
        self.double = double
        if self.half:
            self.model.half()
        if self.double:
            self.model.double()
        if cuda:
            self.model.cuda()

    def fit(self, train_loader, lr=0.01, weight_decay=0, epoch=None, adv=None, optimizer='adam',
            samplings=1, ratio=0.0, lambda2=0.0):
        if type(optimizer) is str:
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=0.9, weight_decay=weight_decay)
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
            else:
                raise ValueError("Optimizer {} not valid.".format(optimizer))
        logging.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss, prec, IBss = train(train_loader, self.model, optimizer, epoch, self.N, half=self.half, double=self.double,
                           vi=self.vi, samplings=samplings, ratio=ratio, lambda2=lambda2)

        return loss, prec, IBss

    def validate(self, val_loader, sample=True):
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()
        loss, prec = validate(val_loader, self.model, criterion, half=self.half, double=self.double, vi=self.vi,
                              sample=sample)
        return loss, prec

    def sample_predict(self, x, Nsamples):
        self.model.eval()
        with torch.no_grad():
            predictions = x.data.new(Nsamples, x.shape[0], self.num_classes)

            Hs = []
            for i in range(Nsamples):
                y, kl = self.model(x)
                predictions[i] = y

                output = nn.functional.softmax(y)
                H = torch.distributions.Categorical(probs=output).entropy()
                Hs.append(H)

            Ha = sum(Hs) / Nsamples
            He = sum(torch.abs(Ha - i) for i in Hs) / Nsamples
        return predictions, Ha, He


class LeNetVIWrapper(NetWrapper):
    def __init__(self, N, half=False, cuda=True, double=False, num_classes=10, vi=True, sigma0=0.15,
                 inits=math.log(0.15)):
        super(LeNetVIWrapper).__init__()
        self.N = N
        self.vi = vi
        self.num_classes = num_classes
        if not vi:
            raise NotImplementedError
        else:
            self.model = Lenet_vi(sigma_0=sigma0, N=N, init_s=inits, nclass=num_classes)
        self.half = half
        self.double = double
        if self.half:
            self.model.half()
        if self.double:
            self.model.double()
        if cuda:
            self.model.cuda()

    def fit(self, train_loader, lr=0.01, weight_decay=0, epoch=None, adv=None, optimizer='adam',
            samplings=1, ratio=0.0, lambda2=0.0):
        if type(optimizer) is str:
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=0.9, weight_decay=weight_decay)
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
            else:
                raise ValueError("Optimizer {} not valid.".format(optimizer))
        logging.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss, prec = train(train_loader, self.model, optimizer, epoch, self.N, half=self.half, double=self.double,
                           vi=self.vi, samplings=samplings, ratio=ratio, lambda2=lambda2)

        return loss, prec

    def validate(self, val_loader, sample=True):
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()
        loss, prec = validate(val_loader, self.model, criterion, half=self.half, double=self.double, vi=self.vi,
                              sample=sample)
        return loss, prec

    def sample_predict(self, x, Nsamples):
        self.model.eval()
        with torch.no_grad():
            predictions = x.data.new(Nsamples, x.shape[0], self.num_classes)

            Hs = []
            for i in range(Nsamples):
                y, kl = self.model(x)
                predictions[i] = y

                output = nn.functional.softmax(y)
                H = torch.distributions.Categorical(probs=output).entropy()
                Hs.append(H)

            Ha = sum(Hs) / Nsamples
            He = sum(torch.abs(Ha - i) for i in Hs) / Nsamples
        return predictions, Ha, He


class VibNetWrapper(NetWrapper):
    def __init__(self):
        super(VibNetWrapper).__init__()

    def fit(self, train_loader, lr=0.01, weight_decay=0, epoch=None, adv=None, optimizer='adam',
            samplings=1, asratio=0.0):
        if type(optimizer) is str:
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=0.9, weight_decay=weight_decay)
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
            else:
                raise ValueError("Optimizer {} not valid.".format(optimizer))
        logging.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss, prec = self.train(train_loader, self.model, optimizer, epoch, self.N, half=self.half, double=self.double,
                                vi=self.vi, samplings=samplings, asratio=asratio)

        return loss, prec

    def validate(self, val_loader, sample=True):
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()
        loss, prec = self.val(val_loader, self.model, criterion, half=self.half, double=self.double, vi=self.vi,
                              sample=sample)
        return loss, prec

    def sample_predict(self, x, Nsamples):
        self.model.eval()
        with torch.no_grad():
            predictions = x.data.new(Nsamples, x.shape[0], self.num_classes)

            Hs = []
            for i in range(Nsamples):
                y, kl = self.model(x)
                predictions[i] = y

                output = nn.functional.softmax(y)
                H = torch.distributions.Categorical(probs=output).entropy()
                Hs.append(H)

            Ha = sum(Hs) / Nsamples
            He = sum(torch.abs(Ha - i) for i in Hs) / Nsamples
        return predictions, Ha, He

    def train(self, train_loader, model, optimizer, epoch, N, half=True, print_freq=50, double=False, vi=True,
              samplings=1, asratio=0.0):
        """
            Run one train epoch
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        model.train()
        end = time.time()
        iters = 5
        alpha = 0.02

        loss, prec1 = None, None
        for i, (input_data, target) in enumerate(train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda()
            input_var = input_data.cuda()
            if half:
                input_var = input_var.half()
            if double:
                input_var = input_var.double()

            # compute output
            if vi:
                loss_sum = 0
                for s in range(samplings):
                    output, kl = model(input_var)
                    tloss = elbo(output, target, kl, get_beta(epoch, N))
                    loss_sum += tloss
                loss = loss_sum / samplings

                output, kl = model(input_var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                if asratio > 1e-7:
                    model(input_var)
                    with torch.no_grad():
                        model.eps.normal_()
                        model.eps.requires_grad = True
                    for index in range(iters):
                        model.eps.requires_grad = True
                        routput, rkl = model(input_var)
                        model.zero_grad()
                        rloss = - F.cross_entropy(routput, target)
                        rloss.backward()
                        with torch.no_grad():
                            model.eps -= alpha * model.eps.grad.sign()
                    routput, rkl = model(input_var)
                    rloss = elbo(routput, target, rkl, get_beta(epoch, N))
                    with torch.no_grad():
                        model.eps = None
                else:
                    rloss = 0
                if 1 - asratio > 1e-7:
                    output, kl = model(input_var)
                    with torch.no_grad():
                        model.eps = None
                    lloss = elbo(output, target, kl, get_beta(epoch, N))
                else:
                    lloss = 0
                loss = asratio * rloss + (1 - asratio) * lloss
                output, kl = model(input_var)
                with torch.no_grad():
                    model.eps = None
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # compute gradient and do SGD step

            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input_data.size(0))
            top1.update(prec1.item(), input_data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                logging.info('Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))
        return losses.avg, top1.avg

    def val(self, val_loader, model, criterion, half=True, print_freq=50, double=False, vi=True, fix=False,
            sample=True, ratio=0.0):
        """
        Run evaluation
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        with torch.no_grad():
            for i, (input_data, target) in enumerate(val_loader):
                target = target.cuda()
                input_var = input_data.cuda()
                target_var = target.cuda()

                if half:
                    input_var = input_var.half()
                if double:
                    input_var = input_var.double()

                # compute output
                if vi:
                    output, kl = model(input_var)
                else:
                    output, kl = model(input_var)
                with torch.no_grad():
                    model.eps = None
                loss = criterion(output, target_var)

                output = output.float()
                loss = loss.float()

                # measure accuracy and record loss
                prec1 = accuracy(output.data, target)[0]
                losses.update(loss.item(), input_data.size(0))
                top1.update(prec1.item(), input_data.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    logging.info('Test: [{0}/{1}]\t'
                                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1))

        logging.info(' * Prec@1 {top1.avg:.3f}'
                     .format(top1=top1))

        return losses.avg, top1.avg


class ToyNetWrapper(VibNetWrapper):
    def __init__(self, N, half=False, cuda=True, double=False, num_classes=10, vi=True, sigma0=0.15,
                 inits=math.log(0.15), mode="vib"):
        super(ToyNetWrapper).__init__()
        self.mode = mode
        self.N = N
        self.vi = vi
        self.num_classes = num_classes
        if not vi:
            self.model = ToyNet(N=N, n_class=num_classes, mode=mode)
        else:
            assert mode == "bnnvib"
            self.model = ToyNet(N=N, n_class=num_classes, mode="bnnvib")
        self.half = half
        self.double = double
        if self.half:
            self.model.half()
        if self.double:
            self.model.double()
        if cuda:
            self.model.cuda()
