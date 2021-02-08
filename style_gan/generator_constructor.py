import numpy as np
import tensorflow as tf

from tensorflow.keras import backend
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model

from style_gan.layers import Constant
from style_gan.layers import NoiseModulation
from style_gan.layers import AdaptiveInstanceModulation


class GeneratorConstructor:

    def __init__(self, **network_config):
        # 1. Resolution related fields.
        self.input_res = network_config['input_res']
        self.output_res = network_config['output_res']
        self.output_res_log2 = int(np.log2(self.output_res))
        self.n_styles = int(self.output_res_log2 * 2 - 2)
        self.n_blocks = self.n_styles // 2
        # 2. Other fields.
        self.latent_size = network_config['latent_size']
        self.n_base_filters = network_config['n_base_filters']
        self.n_max_filters = network_config['n_max_filters']
        self.n_dense_layers = network_config['n_dense_layers']
        self.n_dense_units = network_config['n_dense_units']

    def run(self):
        z_latent, w_latent = self._construct_mapping_network()
        x = self._construct_first_block(w_latent)
        return Model(z_latent, x)

    def _construct_first_block(self, w_latent):
        n_filters = self._filters_at_stage(1)
        x = Constant(shape=(1, self.input_res, self.input_res, n_filters))(w_latent)
        x = NoiseModulation()(x)
        x = LeakyReLU(0.20)(x)
        y = Dense(units=2*n_filters)(w_latent[:, 0, :])
        x = AdaptiveInstanceModulation()([x, y])
        x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
        x = NoiseModulation()(x)
        x = LeakyReLU(0.20)(x)
        y = Dense(units=2*n_filters)(w_latent[:, 1, :])
        x = AdaptiveInstanceModulation()([x, y])
        return x

    def _construct_mapping_network(self):
        z_latent = Input(shape=(self.latent_size,), name="mn_latent")
        x = self._add_mapping_layers(z_latent)
        w_latent = self._broadcast_disentangled_latents(x)
        return z_latent, w_latent

    def _add_mapping_layers(self, x):
        for k in range(self.n_dense_layers):
            x = Dense(units=self.n_dense_units, activation=LeakyReLU(0.20), name="mn_fc_%s" % str(k + 1))(x)
        return x

    def _broadcast_disentangled_latents(self, x):
        return backend.tile(x[:, np.newaxis], [1, self.n_styles, 1])

    def _filters_at_stage(self, stage):
        return np.minimum(int(self.n_base_filters/(2.0**stage)), self.n_max_filters)


config = {
    'input_res': 4,
    'output_res': 1024,
    'latent_size': 512,
    'n_base_filters': 8192,
    'n_max_filters': 512,
    'n_dense_layers': 8,
    'n_dense_units': 512
}

test = GeneratorConstructor(**config).run()
print(test.summary())
