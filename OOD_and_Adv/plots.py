import torch
import matplotlib.pyplot as plt

rootpath = "/slstore/tianfeng/IB_BNN/results/advdataset_seed42_l0.01_epoch200_datasetCIFAR-10_modelvgg_samplings1_ratio0.0_inits-1.8971199848858813/metrics/IBss/"

alldata = []
for i in range(200):
    path = rootpath + f'{i}.pt'
    data = torch.load(path)
    data = torch.vstack(data)
    alldata.append(data)

alldata = torch.stack(alldata, dim=0)  # 200 * 50000 * 12

n_layers = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
plt.plot(alldata.mean(dim=1))
plt.legend([f"layer {n_layers[i]}" for i in range(12)])
plt.xlabel("epoch")
plt.ylabel("Information Bound")
plt.title("Average of IB of all figures of different epochs in origin dataset")
plt.show()

selected_epochs = [0, 40, 80, 150]
for e in selected_epochs:
    plt.hist(alldata[e, :, 6].numpy())
    plt.xlabel("Information Bound")
    plt.ylabel("Frequency number")
    plt.title(f"Distribution at epoch {e}")
    plt.show()