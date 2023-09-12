import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import pickle

def plot(pklpath, savepath):
    with open(pklpath, 'rb') as file:
        res = pickle.load(file)
    plt.clf()
    plt.style.use('ggplot')
    plt.plot(res)
    plt.title(pklpath)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.savefig(savepath)

def plot_log(logpath, savepath):
    kls = []
    with open(logpath, 'r') as l:
        for line in l:
            if "DEBUG" in line and "tensor" not in line:
                ls = line.split(" - ")
                kl = ls[2].split(",")[0]
                kl = float(kl)
                kls.append(kl)

    plt.clf()
    plt.style.use('ggplot')
    plt.yscale("log")
    plt.plot(kls)
    ratio = logpath.find("ratio")
    plt.title(logpath[ratio: ratio + 20])
    plt.xlabel("batch")
    plt.ylabel("Information Bound")
    plt.savefig(savepath)


if __name__ == '__main__':
    paths = [
    "/slstore/tianfeng/IB_BNN/figures/CIFAR-100_vgg/baseline/test_seed42_l0.01_epoch200_datasetCIFAR-100_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/",
    "/slstore/tianfeng/IB_BNN/figures/CIFAR-10_vgg/new_seed42_l0.01_epoch200_datasetCIFAR-10_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/"
]
    for path in paths:
        plot_log(path + "log.log", path + "figures/output_kl.png")


