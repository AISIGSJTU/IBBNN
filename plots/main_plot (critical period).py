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


def plot_log(path, savepath, scale=1):
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (6.0, 4.5)
    kls = []
    logpath = path + 'log.log'
    with open(logpath, 'r') as l:
        for line in l:
            if "DEBUG" in line and "tensor" not in line:
                ls = line.split(" - ")
                kl = ls[2].split(",")[0]
                kl = float(kl)
                kls.append(kl)
    with open(path + 'metrics/test_precs.pkl', 'rb') as file:
        test_precs = pickle.load(file)
    with open(path + 'metrics/train_precs.pkl', 'rb') as file:
        train_precs = pickle.load(file)

    kls = [kls[391 * i] / scale for i in range(200)]
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    klp = ax.plot(kls)
    plt.ylabel("Information Bound")
    plt.xlabel("Epoch")
    # plt.legend(["Information Bound"])
    ax2 = ax.twinx()
    # plt.twiny()
    testp = ax2.plot(test_precs, color='#008888', ls='--')
    trainp = ax2.plot(train_precs, color='#000088', ls='--')
    plt.ylabel("Accuracy")
    ax.legend(klp + testp + trainp, ["Information Bound", "Test Accuracy", "Train Accuracy"], loc=4)

    # plt.title("CIFAR-100")

    plt.savefig(savepath)


def plot_log2(path, savepath, scale=1):
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (6.0, 4.5)
    kls = []
    logpath = path + 'log.log'
    with open(logpath, 'r') as l:
        for line in l:
            if "DEBUG" in line and "tensor" not in line and "var" not in line and "font" not in line:
                ls = line.split(" - ")
                kl = ls[2].split(",")[0]
                kl = float(kl)
                kls.append(kl)
    with open(path + 'metrics/test_precs.pkl', 'rb') as file:
        test_precs = pickle.load(file)
    with open(path + 'metrics/train_precs.pkl', 'rb') as file:
        train_precs = pickle.load(file)

    kls = [kls[391 * i] / scale for i in range(200)]
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    klp = ax.plot(kls)
    plt.ylabel("Information Bound")
    plt.xlabel("Epoch")
    # plt.legend(["Information Bound"])
    ax2 = ax.twinx()
    # plt.twiny()
    testp = ax2.plot(test_precs, color='#008888', ls='--')
    trainp = ax2.plot(train_precs, color='#000088', ls='--')
    plt.ylabel("Accuracy")
    ax.legend(klp + testp + trainp, ["Information Bound", "Test Accuracy", "Train Accuracy"], loc=4)

    # plt.title("CIFAR-100")

    plt.savefig(savepath)


if __name__ == '__main__':
    # path =  "/slstore/tianfeng/IB_BNN/figures/CIFAR-100_vgg/baseline/test_seed42_l0.01_epoch200_datasetCIFAR-100_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/"
    # plot_log(path, "/slstore/tianfeng/IB_BNN/figures/critical_period_C100.pdf", scale=16384)
    #
    # path = "/slstore/tianfeng/IB_BNN/figures/CIFAR-10_vgg/baseline/new_seed42_l0.01_epoch200_datasetCIFAR-10_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/"
    # plot_log(path, "/slstore/tianfeng/IB_BNN/figures/critical_period_C10.pdf")
    #
    path = "/slstore/tianfeng/IB_BNN/figures/FashionMNIST_lenet/baselines/lenet_seed42_l0.01_epoch200_datasetFashionMNIST_modellenet_samplings1_ratio0.0_inits-1.8971199848858813/"
    plot_log2(path, "/slstore/tianfeng/IB_BNN/figures/critical_period_FM.pdf")

    path = "/slstore/tianfeng/IB_BNN/figures/KMNIST_lenet/baseline/lenet_seed42_l0.01_epoch200_datasetKMNIST_modellenet_samplings1_ratio0.0_inits-1.8971199848858813/"
    plot_log2(path, "/slstore/tianfeng/IB_BNN/figures/critical_period_KM.pdf")
