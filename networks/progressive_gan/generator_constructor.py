import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from networks.progressive_gan.generator import Generator
from networks.layers import PixelNormalization
from networks.layers import WeightedSum


class GeneratorConstructorProgressive:
    """ Constructs a list of progressively growing generator models. """
    def __init__(self, **network_config):
        self.latent_size = network_config['latent_size']
        self.output_res = network_config['output_res']
        self.n_max_filters = network_config['n_max_filters']
        self.n_base_filters = network_config['n_base_filters']
        self.adam_params = network_config['adam_params']
        self.n_blocks = int(np.log2(self.output_res)-1)
        self.kernel_init = RandomNormal()

    def run(self):
        """ Executes the progressive_gan of a generator model list. """
        # 1. Initialize list of generators.
        generators = []
        # 2. Construct and add initial generator.
        init_filters = self._number_of_filters(0)
        init_models = self._construct_initial_model(init_filters)
        generators.append(init_models)
        # 3. Construct and add next generators.
        for k in range(1, self.n_blocks):
            old_model = generators[k - 1][0]
            next_filters = self._number_of_filters(k)
            next_models = self._construct_next_models(old_model, next_filters)
            generators.append(next_models)
        return generators

    def _construct_next_models(self, old_model, filters):
        """ Expands old model for new resolution step. """
        # 1. Get the final layer before the RGB layer.
        end_block_old = old_model.layers[-2].output
        # 2. Apply upsampling to the previous convolutional block.
        upsampling_layer = UpSampling2D()(end_block_old)
        # 3. Construct "tuning" generator.
        end_block_new = self._add_convolutional_layers(upsampling_layer, filters, 2)
        output_layer_new = self._add_to_rgb_layer(end_block_new)
        next_model = Generator(old_model.input, output_layer_new)
        self._compile_model(next_model)
        # 4. Construct "fade-in" generator.
        output_layer_old = self._add_to_rgb_layer(upsampling_layer)
        output_layer_wsum = WeightedSum()([output_layer_old, output_layer_new])
        fade_in_model = Generator(old_model.input, output_layer_wsum)
        self._compile_model(fade_in_model)
        return [next_model, fade_in_model]

    def _construct_initial_model(self, filters):
        """ Constructs the initial generator handling a 4x4 resolution. """
        # 1. Construct input layer.
        input_layer = Input(shape=(self.latent_size,))
        # 2. Map latent space to feature maps aka "(4x4) convolutional layer".
        x = self._add_latent_mapping_layer(input_layer, filters)
        # 4. Add a (3x3) convolutional layer.
        x = self._add_convolutional_layers(x, filters, 1)
        # 5. Add toRGB layer.
        output_layer = self._add_to_rgb_layer(x)
        # 6. Construct Keras model.
        initial_model = Generator(input_layer, output_layer)
        self._compile_model(initial_model)
        return [initial_model, initial_model]

    def _add_latent_mapping_layer(self, x, filters):
        """ Maps the latent space to feature maps aka (4x4) convolutional layer. """
        x = Dense(units=4*4*filters, kernel_initializer=self.kernel_init)(x)
        x = Reshape(target_shape=(4, 4, filters))(x)
        x = LeakyReLU(0.20)(x)
        x = PixelNormalization()(x)
        return x

    def _add_to_rgb_layer(self, x):
        """ Adds a (1x1) convolutional layer to generate an RGB image. """
        x = Conv2D(filters=3, kernel_size=(1, 1), padding='same', kernel_initializer=self.kernel_init)(x)
        return x

    def _add_convolutional_layers(self, x, filters, n_layers):
        """ Adds two (3x3) convolutional layers. """
        for _ in range(n_layers):
            x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', kernel_initializer=self.kernel_init)(x)
            x = LeakyReLU(0.20)(x)
            x = PixelNormalization()(x)
        return x

    def _number_of_filters(self, stage):
        return int(np.minimum(self.n_base_filters / (2 ** (stage + 1)), self.n_max_filters))

    def _compile_model(self, model):
        """ Compiles a model using default settings. """
        lr = self.adam_params["lr"]
        beta_1 = self.adam_params["beta_1"]
        beta_2 = self.adam_params["beta_2"]
        epsilon = self.adam_params["epsilon"]
        model.compile(optimizer=Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon))
