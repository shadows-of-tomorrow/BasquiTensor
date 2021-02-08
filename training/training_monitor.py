import os
import numpy as np
import matplotlib.pyplot as plt

from gans.utils import generate_fake_samples
from gans.utils import generate_real_samples


class TrainingMonitor:
    """ Monitors the training performance. """
    def __init__(self, image_processor):
        self.image_processor = image_processor
        self.n_samples = 25
        assert np.sqrt(self.n_samples) % 2 == 1  # This should be an uneven number.

    def store_losses(self, res, fade_in, **losses):
        """ Prints the current resolution and losses to the terminal. """
        dir_out = self.image_processor.dir_out
        # Construct loss message.
        message = f"Resolution:{res},"
        message += f"Fade-in:{fade_in},"
        for key, value in losses.items():
            message += f"{key}:{value},"
        message += "\n"
        # Create dir if non-existent.
        file_dir = dir_out + '/loss.txt'
        if os.path.exists(file_dir):
            mode = 'a'
        else:
            mode = 'w'
        with open(file_dir, mode) as file:
            file.write(message)
            file.close()

    def store_plots(self, generator, step, fade_in):
        """ Generates and stores plots based on current generator. """
        # 1. Extract relevant fields.
        dir_out = os.path.join(self.image_processor.dir_out, 'plots')
        latent_dim = generator.input.shape[1]
        res = generator.output.shape[1]
        # 2. Generate and scale fake images.
        n_fake = int(np.floor(self.n_samples/2))
        x_fake, _ = generate_fake_samples(generator, latent_dim, n_fake)
        x_fake = self.image_processor.shift_arr(x_fake, min_max=True)
        x_fake = [x_fake[k] for k in range(len(x_fake))]
        # 3. Generate and scale real images.
        n_real = int(np.floor(self.n_samples/2))
        x_real, _ = generate_real_samples(self.image_processor, n_real, (res, res))
        x_real = self.image_processor.shift_arr(x_real, min_max=True)
        x_real = [x_real[k] for k in range(len(x_real))]
        # 4. Generate "border" images.
        n_border = int(np.sqrt(self.n_samples))
        x_border = np.zeros((n_border, res, res))
        x_border = [x_border[k] for k in range(len(x_border))]
        # 5. Construct image grid.
        n_grid = int(np.sqrt(self.n_samples))
        for k in range(self.n_samples):
            plt.subplot(n_grid, n_grid, k + 1)
            plt.axis('off')
            if k % n_grid == 0 or (k-1) % n_grid == 0:
                plt.imshow(x_fake.pop())
            elif (k-2) % n_grid == 0:
                plt.imshow(x_border.pop())
            else:
                plt.imshow(x_real.pop())
        # 4. Create output directories (if these do not exist).
        if fade_in:
            file_dir = os.path.join(dir_out, f'{res}x{res}_fade_in')
        else:
            file_dir = os.path.join(dir_out, f'{res}x{res}_tuned')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        # 5. Store images.
        file_name = os.path.join(file_dir, '%s.png' % step)
        plt.suptitle("Fake / Real")
        plt.savefig(file_name)
        plt.close()

    def store_networks(self, discriminator, generator, composite, fade_in):
        """ Stores Keras networks for later use. """
        # 1. Extract relevant fields.
        dir_out = os.path.join(self.image_processor.dir_out, 'networks')
        res = generator.output.shape[1]
        # 2. Create output directories (if these do not exist).
        if fade_in:
            file_dir = os.path.join(dir_out, f'{res}x{res}_fade_in')
        else:
            file_dir = os.path.join(dir_out, f'{res}x{res}_tuned')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        # 3. Store networks.
        discriminator.save(os.path.join(file_dir, f"discriminator.h5"))
        generator.save(os.path.join(file_dir, f"generator.h5"))
        composite.save(os.path.join(file_dir, f"composite.h5"))
