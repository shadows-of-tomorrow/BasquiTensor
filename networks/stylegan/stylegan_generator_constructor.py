import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D
from networks.layers import Constant
from networks.layers import DenseEQL
from networks.layers import Conv2DEQL
from networks.layers import NoiseModulation
from networks.layers import AdaptiveInstanceModulation
from networks.stylegan.stylegan_generator import StyleGANGenerator


class StyleGANGeneratorConstructor:

    def __init__(self, **network_config):
        # 1. Resolution related fields.
        self.output_res = network_config['output_res']
        self.output_res_log2 = int(np.log2(self.output_res))
        self.n_styles = int(self.output_res_log2 * 2 - 2)
        self.n_blocks = self.n_styles // 2
        # 2. Other fields.
        self.latent_size = network_config['latent_size']
        self.latent_dist = network_config['latent_dist']
        self.n_base_filters = network_config['n_base_filters']
        self.n_max_filters = network_config['n_max_filters']
        self.n_mapping_layers = network_config['stylegan_params']['n_mapping_layers']
        self.adam_params = network_config['adam_params']
        self.loss_type = network_config['loss_type']
        self.relu_slope = 0.20

    def run(self):
        # 1. Construct mapping network.
        backend.set_floatx('float32')
        z_latent, w_latent = self._construct_mapping_network()
        # 3. Construct initial block.
        x = self._construct_initial_block(w_latent)
        y = self._add_to_rgb(x, None)
        # 4. Construct and add next blocks.
        for stage in range(2, self.n_blocks + 1):
            x = self._add_next_block(x, w_latent, stage)
            y = UpSampling2D()(y)
            y = self._add_to_rgb(x, y)
        # 5. Construct and compile generator.
        generator = StyleGANGenerator(z_latent, y)
        generator.loss_type = self.loss_type
        generator.latent_dist = self.latent_dist
        self._compile_model(generator)
        return [[generator, None]]

    def _construct_mapping_network(self):
        z_latent = Input(shape=(self.latent_size,), name="mn_latent")
        x = self._add_mapping_layers(z_latent)
        w_latent = self._broadcast_disentangled_latents(x)
        return z_latent, w_latent

    def _construct_initial_block(self, w_latent):
        n_filters = self._compute_filters_at_stage(1)
        # 1. Constant layer + noise.
        x = Constant(shape=(1, 4, 4, n_filters), initializer='ones')(w_latent)
        x = NoiseModulation()(x)
        x = LeakyReLU(self.relu_slope)(x)
        # 2. First AdaIN block.
        y = DenseEQL(units=2 * n_filters)(w_latent[:, 0, :])
        x = AdaptiveInstanceModulation()([x, y])
        # 3. Conv (3x3) layer + noise.
        x = Conv2DEQL(n_channels=n_filters, kernel_size=3)(x)
        x = NoiseModulation()(x)
        x = LeakyReLU(self.relu_slope)(x)
        # 4. Second AdaIN block.
        y = DenseEQL(units=2 * n_filters)(w_latent[:, 1, :])
        x = AdaptiveInstanceModulation()([x, y])
        return x

    def _add_next_block(self, x, w_latent, stage):
        n_filters = self._compute_filters_at_stage(stage)
        # 1. Double resolution operation.
        x = UpSampling2D()(x)
        # 2. First conv (3x3) layer + noise.
        x = Conv2DEQL(n_channels=n_filters, kernel_size=3)(x)
        x = NoiseModulation()(x)
        x = LeakyReLU(self.relu_slope)(x)
        # 3. First AdaIN block.
        y = DenseEQL(units=2 * n_filters)(w_latent[:, 2 * (stage - 1), :])
        x = AdaptiveInstanceModulation()([x, y])
        # 4. Second conv (3x3) layer + noise.
        x = Conv2DEQL(n_channels=n_filters, kernel_size=3)(x)
        x = NoiseModulation()(x)
        x = LeakyReLU(self.relu_slope)(x)
        # 5. Second AdaIN block.
        y = DenseEQL(units=2 * n_filters)(w_latent[:, 2 * (stage - 1) + 1, :])
        x = AdaptiveInstanceModulation()([x, y])
        return x

    def _compile_model(self, model):
        lr = self.adam_params["lr"]
        beta_1 = self.adam_params["beta_1"]
        beta_2 = self.adam_params["beta_2"]
        epsilon = self.adam_params["epsilon"]
        model.compile(optimizer=Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon))

    # ------------------------------------------- Utils ----------------------------------------------------------------
    @staticmethod
    def _add_to_rgb(x, y):
        x = Conv2DEQL(n_channels=3, kernel_size=1)(x)
        if y is not None:
            x += y
        return x

    def _add_mapping_layers(self, x):
        for k in range(self.n_mapping_layers):
            x = DenseEQL(units=self.latent_size, lrmul=0.01)(x)
            x = LeakyReLU(0.20)(x)
        return x

    def _broadcast_disentangled_latents(self, x):
        return backend.tile(x[:, np.newaxis], [1, self.n_styles, 1])

    def _compute_filters_at_stage(self, stage):
        return np.minimum(int(self.n_base_filters / (2.0 ** stage)), self.n_max_filters)
