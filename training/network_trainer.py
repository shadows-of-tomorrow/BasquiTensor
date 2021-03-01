import time
import psutil
from networks.utils import update_fade_in
from networks.utils import update_smoothed_weights
from networks.utils import generate_latent_vectors
from networks.utils import clone_subclassed_model
from training.training_monitor import TrainingMonitor
from processing.augmentation.image_augmenter import ImageAugmenter


class NetworkTrainer:

    def __init__(self, image_processor, **training_config):
        self.stage = 0
        self.image_processor = image_processor
        self.training_monitor = TrainingMonitor(self.image_processor, training_config["monitor_fid"] == "True")
        self.image_augmenter = ImageAugmenter()
        self.n_batches = training_config['n_batches']
        self.n_images = training_config['n_images']
        self.start_time = time.time()

    def run(self, networks):
        # 0. Unpack networks.
        discriminators = networks['discriminators']
        generators = networks['generators']
        assert len(discriminators) == len(generators)
        assert len(discriminators) == len(self.n_batches) == len(self.n_images)
        # 1. Extract initial models.
        discriminator = discriminators[0][0]
        generator = generators[0][0]
        # 2. Train initial models.
        res = generator.output.shape[1]
        print(f"Training gans at {res}x{res} resolution...")
        self._train_epochs(generator, discriminator, self.n_images[0], self.n_batches[0], False)
        # 3. Train models at each growth stage.
        for k in range(1, len(discriminators)):
            # 3.1 Get normal and fade in models.
            [dis_tuning, dis_fade_in] = discriminators[k]
            [gen_tuning, gen_fade_in] = generators[k]
            # 3.2 Train fade-in models.
            res = gen_tuning.output.shape[1]
            print(f"Training gans at {res}x{res} resolution...")
            self._train_epochs(gen_fade_in, dis_fade_in, self.n_images[k], self.n_batches[k], True)
            # 3.3 Train tuning models.
            self._train_epochs(gen_tuning, dis_tuning, self.n_images[k], self.n_batches[k], False)

    def _train_epochs(self, generator, discriminator, n_images, n_batch, fade_in):
        # 1. Compute number of training steps.
        n_steps = n_images // n_batch
        # 2. Get shape of image.
        latent_dim = generator.input.shape[1]
        shape = tuple(generator.output.shape[1:-1].as_list())
        res = shape[0]
        # 3. Clone generator.
        smoothed_generator = clone_subclassed_model(generator)
        # 3. Train models for n_steps iterations.
        for k in range(n_steps):
            # 3.0 Update checks.
            time_checkpoint = time.time()
            # 3.1 Update alpha for weighted sum.
            if fade_in:
                update_fade_in([generator, discriminator], k, n_steps)
            # 3.3 Train discriminator.
            d_loss = discriminator.train_on_batch(self.image_processor, generator, n_batch, shape, self.image_augmenter)
            # 3.4 Train generator.
            z_latent = generate_latent_vectors(latent_dim, n_batch, distribution=generator.latent_dist)
            g_loss = generator.train_on_batch(z_latent, discriminator, n_batch, self.image_augmenter)
            # 3.5 Update "smoothed" generator weights.
            update_smoothed_weights(smoothed_generator, generator)
            # 3.6 Compute auxiliary loss statistics.
            time_loss = time.time() - time_checkpoint
            mem_loss = psutil.virtual_memory().percent
            # 3.6 Construct loss dict.
            loss_dict = {**d_loss, **g_loss, **{'time': f"{time_loss}"}, **{'memory': f"{mem_loss}"}}
            # 3.7 Monitor training process.
            done = k == (n_steps-1)
            self.training_monitor.run(discriminator, generator, smoothed_generator, res, fade_in, k, done, **loss_dict)
