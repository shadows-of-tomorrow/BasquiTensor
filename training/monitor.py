import os
import numpy as np
import matplotlib.pyplot as plt
from networks.utils import generate_fake_samples


def store_plots(generator, latent_dim, n_samples, stage, step, fade_in):
    res = 2 ** (stage+2)
    x, _ = generate_fake_samples(generator, latent_dim, n_samples)
    x = (x - x.min()) / (x.max() - x.min())
    n_grid = int(np.sqrt(n_samples))
    for k in range(n_samples):
        plt.subplot(n_grid, n_grid, k + 1)
        plt.axis('off')
        plt.imshow(x[k])
    file_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'out')
    if fade_in:
        file_dir = os.path.join(file_dir, f'{res}x{res}_fade_in')
    else:
        file_dir = os.path.join(file_dir, f'{res}x{res}_normal')
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    file_name = os.path.join(file_dir, '%s.png' % step)
    plt.savefig(file_name)
    plt.close()
