from networks.utils import update_fade_in
from networks.utils import generate_fake_samples
from networks.utils import generate_real_samples
from networks.utils import generate_latent_points
from training.monitor import store_plots


class GANTrainer:
    """ Trains a set of progressively growing GANs. """
    def __init__(self, latent_dim, image_provider, n_batches, n_epochs):
        self.stage = 0
        self.latent_dim = latent_dim
        self.image_provider = image_provider
        self.n_imgs = image_provider.count_imgs()
        self.n_batches = n_batches
        self.n_epochs = n_epochs

    def execute(self, discriminators, generators, composites):
        print(f"Training model at 4x4 resolution!")
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
            print(f"Training model at {2 ** (self.stage + 2)}x{2 ** (self.stage + 2)} resolution!")
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
        shape = tuple(generator.output.shape[1:-1].as_list())
        # 3. Train models for n_steps iterations.
        for i in range(n_steps):
            # 3.1 Update alpha for weighted sum.
            if fade_in:
                update_fade_in([generator, discriminator, composite], i, n_steps)
            # 3.2 Generate real and fake samples.
            x_real, y_real = generate_real_samples(self.image_provider, n_batch, shape)
            x_fake, y_fake = generate_fake_samples(generator, self.latent_dim, n_batch)
            # 3.3 Train discriminator on real and fake examples.
            d_loss_real = discriminator.train_on_batch(x_real, y_real)
            d_loss_fake = discriminator.train_on_batch(x_fake, y_fake)
            # 3.4 Train generator on discriminator score.
            z = generate_latent_points(self.latent_dim, n_batch)
            g_loss = composite.train_on_batch(z, y_real)
            # 3.5 Monitor progress.
            if i % 100 == 0 or i == (n_steps-i):
                print(f"d_loss_real: {round(d_loss_real,3)}, "
                      f"d_loss_fake: {round(d_loss_fake,3)}, "
                      f"g_loss: {round(g_loss,3)}")
                store_plots(
                    generator=generator,
                    latent_dim=self.latent_dim,
                    n_samples=25,
                    stage=self.stage,
                    step=n_batch*i,
                    fade_in=fade_in
                )
