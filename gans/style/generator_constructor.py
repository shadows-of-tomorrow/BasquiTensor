import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D
from gans.layers import Constant
from gans.layers import NoiseModulation
from gans.layers import AdaptiveInstanceModulation
from gans.style.generator import Generator


class GeneratorConstructorStyle:

    def __init__(self, **network_config):
        # 1. Resolution related fields.
        self.output_res = network_config['output_res']
        self.output_res_log2 = int(np.log2(self.output_res))
        self.n_styles = int(self.output_res_log2 * 2 - 2)
        self.n_blocks = self.n_styles // 2
        # 2. Other fields.
        self.latent_size = network_config['latent_size']
        self.n_base_filters = network_config['n_base_filters']
        self.n_max_filters = network_config['n_max_filters']
        self.n_dense_layers = network_config['n_dense_layers']
        # 3. Kernel initialization.
        self.adam_params = network_config['adam_params']
        self.kernel_init = HeNormal()

    def run(self):
        # 1. Construct mapping network.
        z_latent, w_latent = self._construct_mapping_network()
        # 3. Construct initial block.
        x = self._construct_initial_block(w_latent)
        y = self._add_to_rgb(x, None)
        # 4. Construct and add next blocks.
        for stage in range(2, self.n_blocks+1):
            x = self._add_next_block(x, w_latent, stage)
            y = UpSampling2D()(y)
            y = self._add_to_rgb(x, y)
        # 5. Construct and compile generator.
        generator = Generator(z_latent, y)
        self._compile_model(generator)
        return [[generator, generator]]

    def _construct_initial_block(self, w_latent):
        n_filters = self._compute_filters_at_stage(1)
        # 1. Constant layer + noise.
        x = Constant(shape=(1, 4, 4, n_filters), initializer='ones')(w_latent)
        x = NoiseModulation(activation=LeakyReLU(0.20))(x)
        # 2. First AdaIN block.
        y = Dense(units=2*n_filters, kernel_initializer=self.kernel_init)(w_latent[:, 0, :])
        x = AdaptiveInstanceModulation()([x, y])
        # 3. Conv (3x3) layer + noise.
        x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', kernel_initializer=self.kernel_init)(x)
        x = NoiseModulation(activation=LeakyReLU(0.20))(x)
        # 4. Second AdaIN block.
        y = Dense(units=2*n_filters, kernel_initializer=self.kernel_init)(w_latent[:, 1, :])
        x = AdaptiveInstanceModulation()([x, y])
        return x

    def _add_next_block(self, x, w_latent, stage):
        n_filters = self._compute_filters_at_stage(stage)
        # 1. Double resolution operation.
        x = UpSampling2D()(x)
        # 2. First conv (3x3) layer + noise.
        x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', kernel_initializer=self.kernel_init)(x)
        x = NoiseModulation(activation=LeakyReLU(0.20))(x)
        # 3. First AdaIN block.
        y = Dense(units=2*n_filters, kernel_initializer=self.kernel_init)(w_latent[:, 2*(stage-1), :])
        x = AdaptiveInstanceModulation()([x, y])
        # 4. Second conv (3x3) layer + noise.
        x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', kernel_initializer=self.kernel_init)(x)
        x = NoiseModulation(activation=LeakyReLU(0.20))(x)
        # 5. Second AdaIN block.
        y = Dense(units=2*n_filters, kernel_initializer=self.kernel_init)(w_latent[:, 2*(stage-1)+1, :])
        x = AdaptiveInstanceModulation()([x, y])
        return x

    def _compile_model(self, model):
        lr = self.adam_params["lr"]
        beta_1 = self.adam_params["beta_1"]
        beta_2 = self.adam_params["beta_2"]
        epsilon = self.adam_params["epsilon"]
        model.compile(optimizer=Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon))

    @staticmethod
    def _add_to_rgb(x, y):
        x = Conv2D(filters=3, kernel_size=(1, 1))(x)
        if y is not None:
            x += y
        return x

    def _construct_mapping_network(self):
        z_latent = Input(shape=(self.latent_size,), name="mn_latent")
        x = self._add_mapping_layers(z_latent)
        w_latent = self._broadcast_disentangled_latents(x)
        return z_latent, w_latent

    def _add_mapping_layers(self, x):
        for k in range(self.n_dense_layers):
            x = Dense(units=self.latent_size, activation=LeakyReLU(0.20), kernel_initializer=self.kernel_init,
                      name="mn_fc_%s" % str(k + 1))(x)
        return x

    def _broadcast_disentangled_latents(self, x):
        return backend.tile(x[:, np.newaxis], [1, self.n_styles, 1])

    def _compute_filters_at_stage(self, stage):
        return np.minimum(int(self.n_base_filters/(2.0**stage)), self.n_max_filters)