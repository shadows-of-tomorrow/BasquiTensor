import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from networks.utils.sampling import generate_real_images, generate_fake_images
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
        self.network_store_ticks = 100
        self.weight_hist_store_ticks = 100

    def run(self, discriminator, generator, generator_smoothed, res, fade_in, n_step, done, **loss_dict):
        if n_step % self.loss_store_ticks == 0 or done:
            self.store_losses(res, fade_in, **loss_dict)
        if n_step % self.fid_store_ticks == 0 or done:
            self.store_fid(generator_smoothed, fade_in, res)
        if n_step % self.plot_store_ticks == 0 or done:
            self.store_plots(generator_smoothed, n_step, fade_in)
        if n_step % self.network_store_ticks == 0 or done:
            self.store_networks(discriminator, generator, generator_smoothed, fade_in)
        if n_step % self.weight_hist_store_ticks == 0 or done:
            self.store_weight_histograms(discriminator, generator, generator_smoothed, fade_in, n_step)

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
        self._write_message_to_file(file_dir, message)

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
            transform_type="min_max_to_zero_one_eager"
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
        plt.close('all')

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
        discriminator.save(
            filepath=os.path.join(file_dir, f"discriminator.h5"),
            save_format='h5',
            include_optimizer=True
        )
        generator.save(
            filepath=os.path.join(file_dir, f"generator.h5"),
            save_format='h5',
            include_optimizer=True
        )
        generator_smoothed.save(
            filepath=os.path.join(file_dir, "generator_smoothed.h5"),
            save_format='h5',
            include_optimizer=True
        )
        # 4. Write additional network fields to file.
        file_dir = file_dir + '/fields.txt'
        message = f"loss_type:{discriminator.loss_type},"
        message += f"ada_target:{discriminator.ada_target},"
        message += f"ada_smoothing:{discriminator.ada_smoothing},"
        message += f"latent_dist:{generator.latent_dist},\n"
        self._write_message_to_file(file_dir, message)

    def store_weight_histograms(self, discriminator, generator, generator_smoothed, fade_in, n_step):
        # 0. Get (subsample of) network weights.
        discriminator_weights = self._weights_to_list(discriminator.weights)
        generator_weights = self._weights_to_list(generator.weights)
        generator_s_weights = self._weights_to_list(generator_smoothed.weights)
        # 1. Extract relevant fields.
        dir_out = os.path.join(self.image_processor.dir_out, 'weights')
        res = generator.output.shape[1]
        # 2. Create output directories (if these do not exist).
        if fade_in:
            file_dir = os.path.join(dir_out, f'{res}x{res}_fade_in')
        else:
            file_dir = os.path.join(dir_out, f'{res}x{res}_tuned')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        # 3. Construct and store histograms.
        plt.title("Network Weights")
        plt.hist(discriminator_weights, density=True, bins=1000, label='discriminator', histtype='step')
        plt.hist(generator_weights, density=True, bins=1000, label='generator', histtype='step')
        plt.hist(generator_s_weights, density=True, bins=1000, label='generator (s)', histtype='step')
        plt.xlim((-5.0, 5.0))
        plt.ylim((0.0, 1.0))
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(file_dir, f"weight_hist_{n_step}.png"))
        plt.close()

    def _write_message_to_file(self, file_dir, message):
        if os.path.exists(file_dir):
            mode = 'a'
        else:
            mode = 'w'
        with open(file_dir, mode) as file:
            file.write(message)
            file.close()

    def _weights_to_list(self, weights):
        weights = [x.numpy().ravel().tolist() for x in weights]
        weights = [x for sub_list in weights for x in sub_list]
        weights = np.random.choice(weights, size=10000)
        return weights

