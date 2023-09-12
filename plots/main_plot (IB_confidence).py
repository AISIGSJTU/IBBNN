import pickle
import matplotlib.pyplot as plt

def plot(path):
    with open(path + 'retained_acc_by_ib.pkl', 'rb') as file:
        accs = pickle.load(file)
    plt.clf()
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (6.0, 4.5)
    plt.xlabel("Accept Rate")
    plt.ylabel("Accuracy")
    plt.xlim(10, 100)
    plt.plot(range(10, 110, 10), accs, marker="*")



paths = ['/slstore/tianfeng/IB_BNN/figures/CIFAR-100_vgg/baseline/test_seed42_l0.01_epoch200_datasetCIFAR-100_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/',
         '/slstore/tianfeng/IB_BNN/figures/CIFAR-10_vgg/baseline/new_seed42_l0.01_epoch200_datasetCIFAR-10_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/',
         "/slstore/tianfeng/IB_BNN/figures/FashionMNIST_lenet/baselines/lenet_seed42_l0.01_epoch200_datasetFashionMNIST_modellenet_samplings1_ratio0.0_inits-1.8971199848858813/",
         "/slstore/tianfeng/IB_BNN/figures/KMNIST_lenet/baseline/lenet_seed42_l0.01_epoch200_datasetKMNIST_modellenet_samplings1_ratio0.0_inits-1.8971199848858813/",
         ]


plot(paths[0])
plt.savefig("../figures/IBCC100.pdf")
plot(paths[1])
plt.savefig("../figures/IBCC10.pdf")
plot(paths[2])
plt.savefig("../figures/IBCFM.pdf")
plot(paths[3])
plt.savefig("../figures/IBCKM.pdf")
