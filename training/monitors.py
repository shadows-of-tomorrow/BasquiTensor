import os
import numpy as np
import matplotlib.pyplot as plt
from construction.utils import generate_fake_samples


class Monitor:
    """ Monitors the training performance. """
    def __init__(self, image_processor):
        self.image_processor = image_processor
        self.loss_scaling = 10 ** 0
        self.loss_rounding = 10 ** 2
        self.n_samples = 25

    def print_losses(self, res, **losses):
        """ Prints the current resolution and losses to the terminal. """
        message = f"Resolution: {res}x{res}, "
        for key, value in losses.items():
            value = value * self.loss_scaling
            value = int(value * self.loss_rounding) / self.loss_rounding
            message += f"{key}: {value}, "
        print(message)

    def store_plots(self, generator, step, fade_in):
        """ Generates and stores plots based on current generator. """
        # 1. Extract relevant fields.
        dir_out = self.image_processor.dir_out
        latent_dim = generator.input.shape[1]
        res = generator.output.shape[1]
        # 2. Generate and scale fake images.
        x, _ = generate_fake_samples(generator, latent_dim, self.n_samples)
        x = self.image_processor.shift_arr(x, min_max=True)
        # 3. Construct image grid.
        n_grid = int(np.sqrt(self.n_samples))
        for k in range(self.n_samples):
            plt.subplot(n_grid, n_grid, k + 1)
            plt.axis('off')
            plt.imshow(x[k])
        # 4. Create output directories (if these do not exist).
        if fade_in:
            file_dir = os.path.join(dir_out, f'{res}x{res}_fade_in')
        else:
            file_dir = os.path.join(dir_out, f'{res}x{res}_normal')
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        # 5. Store images.
        file_name = os.path.join(file_dir, '%s.png' % step)
        plt.savefig(file_name)
        plt.close()
