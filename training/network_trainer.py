from construction.utils import update_fade_in
from construction.utils import generate_fake_samples
from construction.utils import generate_real_samples
from construction.utils import generate_latent_points
from training.training_monitor import TrainingMonitor


class NetworkTrainer:
    """ Trains a set of progressively growing GANs. """
    def __init__(self, image_processor, **training_config):
        self.stage = 0
        self.image_processor = image_processor
        self.monitor = TrainingMonitor(self.image_processor)
        self.n_imgs = image_processor.count_imgs()
        self.n_batches = training_config['n_batches']
        self.n_epochs = training_config['n_epochs']

    def run(self, networks):
        # 0. Unpack networks.
        discriminators = networks['discriminators']
        generators = networks['generators']
        composites = networks['composites']
        assert len(discriminators) == len(generators) == len(composites)
        assert len(discriminators) == len(self.n_batches) == len(self.n_epochs)
        print(f"Training networks at 4x4 resolution...")
        # 1. Extract initial models.
        init_discriminator = discriminators[0][0]
        init_generator = generators[0][0]
        init_composite = composites[0][0]
        # 2. Train initial models.
        self._train_epochs(
            generator=init_generator,
            discriminator=init_discriminator,
            composite=init_composite,
            n_epoch=self.n_epochs[0],
            n_batch=self.n_batches[0],
            fade_in=False
        )
        # 3. Train models at each growth stage.
        for k in range(1, len(composites)):
            self.stage += 1
            print(f"Training networks at {2 ** (self.stage + 2)}x{2 ** (self.stage + 2)} resolution...")
            # 3.1 Get normal and fade in models.
            [dis_normal, dis_fade_in] = discriminators[k]
            [gen_normal, gen_fade_in] = generators[k]
            [comp_normal, comp_fade_in] = composites[k]
            # 3.2 Train fade-in models.
            self._train_epochs(
                generator=gen_fade_in,
                discriminator=dis_fade_in,
                composite=comp_fade_in,
                n_epoch=self.n_epochs[k],
                n_batch=self.n_batches[k],
                fade_in=True
            )
            # 3.3 Train normal models.
            self._train_epochs(
                generator=gen_normal,
                discriminator=dis_normal,
                composite=comp_normal,
                n_epoch=self.n_epochs[k],
                n_batch=self.n_batches[k],
                fade_in=False
            )

    def _train_epochs(self, generator, discriminator, composite, n_epoch, n_batch, fade_in):
        # 1. Compute number of training steps.
        n_steps = int(self.n_imgs / n_batch) * n_epoch
        # 2 Get shape of image.
        latent_dim = generator.input.shape[1]
        shape = tuple(generator.output.shape[1:-1].as_list())
        # 3. Train models for n_steps iterations.
        for k in range(n_steps):
            # 3.1 Update alpha for weighted sum.
            if fade_in:
                update_fade_in([generator, discriminator, composite], k, n_steps)
            # 3.2 Generate real and fake samples.
            x_real, y_real = generate_real_samples(self.image_processor, n_batch, shape)
            x_fake, y_fake = generate_fake_samples(generator, latent_dim, n_batch)
            # 3.3 Train discriminator on real and fake examples.
            d_loss = discriminator.train_on_batch(x_real, x_fake)
            # 3.4 Train generator on discriminator score.
            z_latent = generate_latent_points(latent_dim, n_batch)
            g_loss = composite.train_on_batch(z_latent, y_fake)
            # 3.5 Monitor progress.
            if k % 100 == 0:
                loss_dict = {**d_loss, **{'g_loss': g_loss}}
                self.monitor.store_losses(shape[0], fade_in, **loss_dict)
                self.monitor.store_plots(generator, n_batch * k, fade_in)
        self.monitor.store_networks(discriminator, generator, composite, fade_in)
