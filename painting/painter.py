import os
import matplotlib.pyplot as plt
from progressive_gan.utils import generate_fake_samples


class Painter:

    def __init__(self, generator, image_processor):
        self.generator = generator
        self.image_processor = image_processor
        self.latent_dim = generator.input_shape[1]

    def paint_to_dir(self, n_paintings, dir_out):
        x_fake = self._generate_paintings(n_paintings)
        self._store_paintings(x_fake, dir_out)

    def _store_paintings(self, paintings, dir_out):
        for k in range(paintings.shape[0]):
            painting = self.image_processor.shift_arr(paintings[k], min_max=True)
            plt.imsave(os.path.join(dir_out, f"{k + 1}.png"), painting)

    def _generate_paintings(self, n_paintings):
        x_fake, _ = generate_fake_samples(self.generator, self.latent_dim, n_paintings)
        return x_fake
