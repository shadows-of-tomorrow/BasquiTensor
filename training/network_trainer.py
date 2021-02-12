import numpy as np
from networks.utils import update_fade_in
from networks.utils import generate_latent_points
from training.training_monitor import TrainingMonitor


class NetworkTrainer:
    """ Trains a set of progressively growing GANs. """

    def __init__(self, image_processor, **training_config):
        self.stage = 0
        self.image_processor = image_processor
        self.monitor = TrainingMonitor(self.image_processor, training_config["monitor_fid"] == "True")
        self.n_imgs = image_processor.count_imgs()
        self.n_batches = training_config['n_batches']
        self.n_epochs = training_config['n_epochs']

    def run(self, networks):
        # 0. Unpack networks.
        discriminators = networks['discriminators']
        generators = networks['generators']
        assert len(discriminators) == len(generators)
        assert len(discriminators) == len(self.n_batches) == len(self.n_epochs)
        # 1. Extract initial models.
        init_discriminator = discriminators[0][0]
        init_generator = generators[0][0]
        # 2. Train initial models.
        res = init_generator.output.shape[1]
        print(f"Training networks at {res}x{res} resolution...")
        self._train_epochs(
            generator=init_generator,
            discriminator=init_discriminator,
            n_epoch=self.n_epochs[0],
            n_batch=self.n_batches[0],
            fade_in=False
        )
        # 3. Train models at each growth stage.
        for k in range(1, len(discriminators)):
            # 3.1 Get normal and fade in models.
            [dis_normal, dis_fade_in] = discriminators[k]
            [gen_normal, gen_fade_in] = generators[k]
            # 3.2 Train fade-in models.
            res = gen_normal.output.shape[1]
            print(f"Training networks at {res}x{res} resolution...")
            self._train_epochs(gen_fade_in, dis_fade_in, self.n_epochs[k], self.n_batches[k], True)
            # 3.3 Train normal models.
            self._train_epochs(gen_normal, dis_normal, self.n_epochs[k], self.n_batches[k], False)

    def _train_epochs(self, generator, discriminator, n_epoch, n_batch, fade_in):
        # 1. Compute number of training steps.
        n_steps = int(self.n_imgs / n_batch) * n_epoch
        # 2 Get shape of image.
        latent_dim = generator.input.shape[1]
        shape = tuple(generator.output.shape[1:-1].as_list())
        # 3. Train models for n_steps iterations.
        for k in range(n_steps):
            # 3.1 Update alpha for weighted sum.
            if fade_in:
                update_fade_in([generator, discriminator], k, n_steps)
            # 3.3 Train discriminator.
            d_loss = discriminator.train_on_batch(self.image_processor, generator, n_batch, shape)
            # 3.4 Train generator.
            z_latent = generate_latent_points(latent_dim, n_batch)
            g_loss = generator.train_on_batch(z_latent, discriminator, n_batch)
            # 3.5 Monitor progress.
            loss_dict = {**d_loss, **g_loss}
            self.monitor.store_losses(shape[0], fade_in, **loss_dict)
            if k % 100 == 0:
                self.monitor.store_fid(shape[0], fade_in, generator)
            if k % 100 == 0:
                self.monitor.store_plots(generator, n_batch * k, fade_in)
        self.monitor.store_networks(discriminator, generator, fade_in)
