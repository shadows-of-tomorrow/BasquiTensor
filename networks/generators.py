import numpy as np
from networks.layers import PixelNormalization, WeightedSum
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.layers import LeakyReLU, UpSampling2D, Dense, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.initializers import HeNormal


class GeneratorCreator:
    """ Creates a list of progressively growing generator models. """
    def __init__(self, latent_dim=256, input_res=4, output_res=256, max_filters=512):
        self.latent_dim = latent_dim
        self.input_res = input_res
        self.output_res = output_res
        self.base_filters = int(2 ** 10)
        self.max_filters = max_filters
        self.n_blocks = int(np.log2(output_res / input_res) + 1)
        self.kernel_init = HeNormal()
        self.kernel_const = max_norm(1.0)

    def execute(self):
        """ Executes the creation of a generator model list. """
        # 1. Initialize list of discriminators.
        generators = []
        # 2. Construct and add initial discriminator.
        init_filters = self._number_of_filters(0)
        init_models = self._construct_init_models(init_filters)
        generators.append(init_models)
        # 3. Construct and add next discriminator.
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
        # 3. Construct "normal" generator.
        end_block_new = self._add_conv_blocks(upsampling_layer, filters)
        output_layer_new = self._add_rgb_layer(end_block_new)
        next_model = Model(old_model.input, output_layer_new)
        # 4. Construct "fade-in" generator.
        output_layer_old = old_model.layers[-1](upsampling_layer)
        output_layer_wsum = WeightedSum()([output_layer_old, output_layer_new])
        fade_in_model = Model(old_model.input, output_layer_wsum)
        return [next_model, fade_in_model]

    def _construct_init_models(self, filters):
        """ Constructs the initial generator handling a 4x4 resolution. """
        # 1. Construct input layer.
        input_layer = Input(shape=(self.latent_dim,))
        # 2. Map latent space to feature maps aka "(4x4) convolutional layer".
        g = Dense(units=self.input_res * self.input_res * filters,
                  kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const)(input_layer)
        g = Reshape(target_shape=(self.input_res, self.input_res, filters))(g)
        g = LeakyReLU(0.20)(g)
        g = PixelNormalization()(g)
        # 4. Add (3x3) convolutional layer.
        g = Conv2D(filters=filters, kernel_size=(3, 3), padding='same',
                   kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const)(g)
        g = LeakyReLU(0.20)(g)
        g = PixelNormalization()(g)
        # 5. Add output (toRGB) layer.
        output_layer = self._add_rgb_layer(g)
        # 6. Properly define model.
        init_model = Model(input_layer, output_layer)
        return [init_model, init_model]

    def _add_rgb_layer(self, x):
        """ Adds a (1x1) convolutional layer. """
        x = Conv2D(filters=3, kernel_size=(1, 1), padding='same',
                   kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const)(x)
        return x

    def _add_conv_blocks(self, x, filters):
        """ Adds two (3x3) convolutional layers. """
        for _ in range(2):
            x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same',
                       kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const)(x)
            x = LeakyReLU(0.20)(x)
            x = PixelNormalization()(x)
        return x

    def _number_of_filters(self, stage):
        return int(np.minimum(self.base_filters / (2**(stage+1)), self.max_filters))
