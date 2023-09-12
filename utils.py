import logging
import sys
import time
import copy
import random
import os

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from models.vgg_vi import VGG_vi
from models.vgg import VGG
from models.resnet import resnet20, resnet56
from models.toynet import ToyNet
from models.lenet_vi import Lenet_vi


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def cprint(color: str, text: str, **kwargs) -> None:
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_beta(epoch_idx, N):
    return 1.0 / N / 100


def elbo(out, y, kl_sum, beta):
    ce_loss = F.cross_entropy(out, y)
    return ce_loss + beta * kl_sum


def train(train_loader, model, optimizer, epoch, N, half=True, print_freq=50, double=False, vi=True, samplings=1,
          ratio=0.0, lambda2=0.0):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    IBss = []

    # switch to train mode
    model.train()
    end = time.time()

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
                if type(model) == Lenet_vi:
                    IB = model.conv_layers[4].kl_output(mean_batch=False)
                elif type(model) == VGG_vi:
                    n_layers = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
                    IBs = torch.vstack([model.features[i].kl_output(mean_batch=False).detach().cpu() for i in n_layers])
                    IBss.append(IBs.transpose(1, 0))

                    IB = IBs[5]
                else:
                    raise NotImplementedError
                IB_mean = IB[5].mean()

                logging.info(f"{IB_mean.item()}, {kl.item()}")
                if ratio > 1e-10:
                    # IB = model.layer2[2].conv2.kl_output()
                    tloss += ratio * IB_mean
                loss_sum += tloss

                t = torch.var(IB)
                logging.info(f"var {t.item()}")
                # print(loss_sum, t, lambda2)
                if lambda2 > 1e-10:
                    loss_sum += lambda2 * t

            loss = loss_sum / samplings

            output, kl = model(input_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:

            output = model(input_var)
            loss = F.cross_entropy(output, target)
            output = model(input_var)
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

    return losses.avg, top1.avg, IBss


def validate(val_loader, model, criterion, half=True, print_freq=50, double=False, vi=True, fix=False, sample=True):
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
                output, kl = model(input_var, fix=fix, sample=sample)
            else:
                output = model(input_var)
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


def data2img(data, path, reshape=True):
    if reshape:
        data = np.einsum('ijk->jki', data)
    data = (data * 255).round()
    data = np.uint8(data)
    img = Image.fromarray(data)
    img.save(path)


def KLD_cost(mu_p, sig_p, mu_q, sig_q):
    KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return KLD


def mean_and_max_de(*mes):
    mean = sum(mes) / len(mes)
    diffs = [abs(mean - i) for i in mes]
    return mean, diffs


if __name__ == '__main__':
    for color in ['a', 'r', 'g', 'y', 'b', 'p', 'c', 'w']:
        cprint(color, color)
        cprint('*' + color, '*' + color)
