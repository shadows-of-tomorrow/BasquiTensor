import numpy as np

from tensorflow.keras import backend
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model


class SynthesisNetworkConstructor:

    def __init__(self, **config):
        self.input_res = config['input_res']
        self.output_res = config['output_res']
        self.latent_size = config['latent_size']
        self.n_base_filters = config['n_base_filters']
        self.n_max_filters = config['n_max_filters']
        self.output_res_log2 = int(np.log2(self.output_res))
        self.n_styles = int(self.output_res_log2 * 2 - 2)
        self._validate_config()

    def run(self):
        input_layer = self._construct_input_layer()

    def _construct_input_layer(self):
        n_filters = self._filters_at_stage(1)
        return Input(shape=(self.input_res, self.input_res, n_filters))

    def _validate_config(self):
        assert 2 ** self.output_res_log2 == self.output_res
        assert self.output_res >= 4

    def _filters_at_stage(self, stage):
        return np.minimum(int(self.n_base_filters / (2.0 ** stage)), self.n_max_filters)


class MappingNetworkConstructor:

    def __init__(self, **config):
        self.latent_size = config['latent_size']
        self.n_dense_layers = config['n_dense_layers']
        self.n_dense_units = config['n_dense_units']
        self.output_res = config['output_res']
        self.output_res_log2 = int(np.log2(self.output_res))
        self.n_styles = int(self.output_res_log2 * 2 - 2)

    def run(self):
        input_layer = Input(shape=(self.latent_size,), name="mn_latent")
        x = self._add_mapping_layers(input_layer)
        output_layer = self._broadcast_disentangled_latents(x)
        return Model(input_layer, output_layer)

    def _add_mapping_layers(self, x):
        for k in range(self.n_dense_layers):
            x = Dense(units=self.n_dense_units, activation=LeakyReLU(0.20), name="mn_fc_%s" % str(k + 1))(x)
        return x

    def _broadcast_disentangled_latents(self, x):
        return backend.tile(x[:, np.newaxis], [1, self.n_styles, 1])
