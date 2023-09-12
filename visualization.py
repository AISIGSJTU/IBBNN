import numpy as np
import matplotlib.pyplot as plt


def to_img(image, path):
    plt.imshow(image, cmap='Greys')
    plt.savefig(path)


if __name__ == '__main__':
   a = np.random.rand(28, 28)
   to_img(a, "tmp.png")