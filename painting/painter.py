import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from networks.utils import generate_latent_vectors_gaussian
from networks.utils import generate_fake_images_from_latents
from networks.layers import Constant
from networks.layers import DenseEQL
from networks.layers import Conv2DEQL
from networks.layers import NoiseModulation
from networks.layers import AdaptiveInstanceModulation
from processing.image_processor import ImageProcessor

CUSTOM_LAYER_MAPPING = {
    "Constant": Constant,
    "DenseEQL": DenseEQL,
    "Conv2DEQL": Conv2DEQL,
    "NoiseModulation": NoiseModulation,
    "AdaptiveInstanceModulation": AdaptiveInstanceModulation
}


class Painter:

    def __init__(self, dir_in, dir_out, gen_name, image_processor):
        self.dir_in = dir_in
        self.dir_out = dir_out
        self.gen_name = gen_name
        self.image_processor = image_processor
        self.generator = self._load_generator(self.gen_name)
        self.base_shape = self.generator.output_shape[1:3]
        self.n_int_steps = 100

    def paint(self, n_paintings, painting_type="basic"):
        x_fake = self._generate_paintings(self.generator, n_paintings, self.base_shape, painting_type)
        self._store_paintings(x_fake)

    def _load_generator(self, gen_name):
        dir_gen = os.path.join(self.dir_in, gen_name)
        return load_model(dir_gen, CUSTOM_LAYER_MAPPING)

    def _store_paintings(self, paintings):
        for k in range(paintings.shape[0]):
            dir_file = os.path.join(self.dir_out, f"{k + 1}.png")
            img = np.clip(paintings[k], 0.0, 1.0)
            plt.imsave(dir_file, img)

    def _generate_paintings(self, generator, n_paintings, shape, painting_type):
        if painting_type == "basic":
            return self._generate_basic_paintings(generator, n_paintings, shape)
        elif painting_type == "interpolated":
            return self._generate_interpolated_paintings(generator, shape, self.n_int_steps)
        else:
            ValueError(f"Painting type {painting_type} not recognized.")

    def _generate_basic_paintings(self, generator, n_paintings, shape):
        z = generate_latent_vectors_gaussian(generator.input_shape[1], n_paintings)
        x = generate_fake_images_from_latents(z, self.image_processor, generator, shape, transform_type='min_max_to_zero_one')
        return x

    def _generate_interpolated_paintings(self, generator, shape, n_int):
        z_pair = generate_latent_vectors_gaussian(generator.input_shape[1], 2)
        z_int = [(k/n_int)*z_pair[0, :] + (1-k/n_int)*z_pair[1, :] for k in range(n_int)]
        z_int = np.stack(z_int, axis=0)
        x = generate_fake_images_from_latents(z_int, self.image_processor, generator, shape, transform_type="min_max_to_zero_one")
        return x


if __name__ == "__main__":
    name = 'bob_ross_256.h5'
    dir_in = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'io', 'input', 'generators')
    dir_out = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'io', 'output', 'painting')
    painter = Painter(dir_in, dir_out, name, ImageProcessor())
    painter.paint(n_paintings=100, painting_type="interpolated")
