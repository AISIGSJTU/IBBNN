import matplotlib.pyplot as plt

x = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 10000]
y = [61.15, 61.45, 63.38, 62.84, 62.81, 63.76, 63.42, 62.67, 63.29, 62.86, 61.63, 59.36, 57.49, 44.25, 29.85]

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (6.0, 4.5)
plt.plot(x, y, marker='*')
plt.xscale('log')

plt.hlines(61.06, 0, 1e4, linestyles="dashed", colors="gray")
plt.xlabel('$\lambda_2$')
plt.ylabel('accuracy')
plt.legend(["Models w. IB Variance Reg.", "Model w/o. IB Variance Reg."])
plt.savefig('../figures/asl2c100.pdf')

x = list(reversed([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])) + [1e-1, 1, 10, 100, 1000]
y = list(reversed([90.60, 90.59, 90.76, 90.49, 90.34])) + [90.36, 91.03, 90.70, 90.06, 69.61]
plt.clf()
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (6.0, 4.5)
plt.plot(x, y, marker='*')
plt.xscale('log')
plt.hlines(90.48, 1e-6, 1e3, linestyles="dashed", colors="gray")
plt.xlabel('$\lambda_2$')
plt.ylabel('accuracy')
plt.legend(["Models w. IB Variance Reg.", "Model w/o. IB Variance Reg."])
plt.savefig('../figures/asl2FM.pdf')

x = [0.0, 1e-9, 1e-8, 1e-7, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1e-1, 1][:-2]
y = [61.15, 61.55, 61.10, 60.91, 62.60, 62.18, 62.15, 61.63, 61.29, 61.67, 61.08, 59.35, 9.15, 1.31][:-2]
plt.clf()
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (6.0, 4.5)
plt.plot(x, y, marker='*')
plt.xscale('log')
# plt.yscale('log')

plt.xlabel('$\lambda_1$')
plt.ylabel('accuracy')
plt.hlines(61.06, 0, 0.01, linestyles="dashed", colors="gray")
plt.legend(["Models w. IB Reg.", "Model w/o. IBReg."])
plt.savefig('../figures/asl1c100.pdf')

x = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
y = [81.68, 90.49, 90.73, 90.40, 90.33]
plt.clf()
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (6.0, 4.5)
plt.plot(x, y, marker='*')
plt.xscale('log')
# plt.yscale('log')

plt.xlabel('$\lambda_1$')
plt.ylabel('accuracy')
plt.hlines(90.48, 1e-6, 0.01, linestyles="dashed", colors="gray")
plt.legend(["Models w. IB Reg.", "Model w/o. IBReg."])
plt.savefig('../figures/asl1FM.pdf')
