import os
import numpy as np
import matplotlib.pyplot as plt
from networks.utils import generate_fake_images
from networks.utils import generate_real_images
from training.fid_calculator import FIDCalculator


class TrainingMonitor:
    """ Monitors the training performance. """

    def __init__(self, image_processor, monitor_fid=True):
        self.n_plot_samples = 25
        self.image_processor = image_processor
        self.monitor_fid = monitor_fid
        if self.monitor_fid:
            self.ffid_calculator = FIDCalculator(self.image_processor)
        self.loss_store_ticks = 1
        self.fid_store_ticks = 100
        self.plot_store_ticks = 100
        self.network_store_ticks = 1000

    def run(self, discriminator, generator, generator_smoothed, res, fade_in, n_step, done, **loss_dict):
        if n_step % self.loss_store_ticks == 0 or done:
            self.store_losses(res, fade_in, **loss_dict)
        if n_step % self.fid_store_ticks == 0 or done:
            self.store_fid(generator_smoothed, fade_in, res)
        if n_step % self.plot_store_ticks == 0 or done:
            self.store_plots(generator_smoothed, n_step, fade_in)
        if n_step % self.network_store_ticks == 0 or done:
            self.store_networks(discriminator, generator, generator_smoothed, fade_in)

    def store_fid(self, generator, res, fade_in):
        if self.monitor_fid:
            fid = self.ffid_calculator.compute_fast_fid(generator)
            message = f"Resolution:{res},Fade-in:{fade_in},FID:{fid},\n"
            file_dir = self.image_processor.dir_out + '/fid.txt'
            if os.path.exists(file_dir):
                mode = 'a'
            else:
                mode = 'w'
            with open(file_dir, mode) as file:
                file.write(message)
                file.close()

    def store_losses(self, res, fade_in, **losses):
        """ Prints the current resolution and losses to the terminal. """
        # 1. Construct loss message.
        message = f"Resolution:{res},"
        message += f"Fade-in:{fade_in},"
        for key, value in losses.items():
            message += f"{key}:{value},"
        message += "\n"
        # 2. Write message to file.
        file_dir = self.image_processor.dir_out + '/loss.txt'
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
        res = generator.output.shape[1]
        # 2. Generate and scale fake images.
        n_fake = int(np.floor(self.n_plot_samples / 2))
        x_fake = generate_fake_images(
            image_processor=self.image_processor,
            generator=generator,
            n_samples=n_fake,
            shape=(res, res),
            transform_type="min_max_to_zero_one"
        )
        x_fake = [np.clip(x_fake[k], 0.0, 1.0) for k in range(len(x_fake))]
        # 3. Generate and scale real images.
        n_real = int(np.floor(self.n_plot_samples / 2))
        x_real = generate_real_images(
            image_processor=self.image_processor,
            n_samples=n_real,
            shape=(res, res),
            transform_type="old_to_zero_one"
        )
        x_real = [x_real[k] for k in range(len(x_real))]
        # 4. Generate "border" images.
        n_border = int(np.sqrt(self.n_plot_samples))
        x_border = np.zeros((n_border, res, res))
        x_border = [x_border[k] for k in range(len(x_border))]
        # 5. Construct image grid.
        n_grid = int(np.sqrt(self.n_plot_samples))
        for k in range(self.n_plot_samples):
            plt.subplot(n_grid, n_grid, k + 1)
            plt.axis('off')
            if k % n_grid == 0 or (k - 1) % n_grid == 0:
                plt.imshow(x_fake.pop())
            elif (k - 2) % n_grid == 0:
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

    def store_networks(self, discriminator, generator, generator_smoothed, fade_in):
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
        generator_smoothed.save(os.path.join(file_dir, "generator_smoothed.h5"))
