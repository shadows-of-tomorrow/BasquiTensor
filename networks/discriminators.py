import numpy as np
from networks.loss_functions import wasserstein_loss_drift_penalty
from networks.layers import WeightedSum, MinibatchStDev
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, LeakyReLU, Dense, Conv2D
from tensorflow.keras.layers import AveragePooling2D, Flatten
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.constraints import max_norm


class DiscriminatorCreator:
    """ Creates a list of progressively growing discriminator models. """
    def __init__(self, skip_layers=2, input_res=4, output_res=256, max_filters=512):
        self.skip_layers = skip_layers
        self.input_res = input_res
        self.output_res = output_res
        self.base_filters = int(2 ** 10)
        self.max_filters = max_filters
        self.n_blocks = int(np.log2(output_res/input_res)+1)
        self.kernel_init = HeNormal()
        self.kernel_const = max_norm(1.0)

    def execute(self):
        """ Executes the creation of a discriminator model list. """
        # 1. Initialize list of discriminators.
        discriminators = []
        # 2. Construct and add initial discriminator.
        init_filters = self._number_of_filters(0)
        init_models = self._construct_init_models(init_filters)
        discriminators.append(init_models)
        # 3. Construct and add next discriminator.
        for k in range(1, self.n_blocks):
            old_model = discriminators[k - 1][0]
            next_models = self._construct_next_models(old_model, k)
            discriminators.append(next_models)
        return discriminators

    def _construct_next_models(self, old_model, stage):
        """ Expands old model for next resolution step. """
        # 1. Double the input resolution of old model.
        input_layer = self._double_input_res(old_model)
        # 2. Add (new) initial block to input layer.
        init_block_new = self._add_new_init_conv_block(input_layer, stage)
        # 3. Add (old) initial block input layer.
        init_block_old = self._add_old_init_conv_block(input_layer, old_model)
        # 4. Compute weighted sum of input blocks.
        init_block_wsum = WeightedSum()([init_block_old, init_block_new])
        # 5. Add old model layers to initial blocks.
        d_next = self._add_old_layers(init_block_new, old_model)
        d_fade_in = self._add_old_layers(init_block_wsum, old_model)
        # 6. Properly define models.
        next_model = Model(input_layer, d_next)
        fade_in_model = Model(input_layer, d_fade_in)
        # 7. Compile models.
        self._compile_model(next_model)
        self._compile_model(fade_in_model)
        return [next_model, fade_in_model]

    def _construct_init_models(self, filters):
        """ Constructs the initial discriminator handling a 4x4 resolution. """
        # 1. Construct input layer.
        input_layer = Input(shape=(self.input_res, self.input_res, 3))
        # 2. Add (1x1) convolutional layer.
        x = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation=LeakyReLU(0.20),
                   kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const)(input_layer)
        # 3. Add minibatch stddev feature map.
        x = MinibatchStDev()(x)
        # 4. Add (3x3) convolutional layer.
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=LeakyReLU(0.20),
                   kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const)(x)
        # 5. Add (4x4) convolutional layer.
        x = Conv2D(filters=filters, kernel_size=(4, 4), padding='same', activation=LeakyReLU(0.20),
                   kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const)(x)
        # 6. Add final dense layer.
        x = Flatten()(x)
        x = Dense(units=1,
                  kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const)(x)
        # 7. Properly define initial model.
        init_model = Model(input_layer, x)
        # 8. Compile initial model.
        self._compile_model(init_model)
        return [init_model, init_model]

    def _add_old_layers(self, new_model, old_model):
        """ Adds the old model layers to the new model (skipping the input layers). """
        for k in range(self.skip_layers, len(old_model.layers)):
            new_model = old_model.layers[k](new_model)
        return new_model

    @staticmethod
    def _add_old_init_conv_block(input_layer, old_model):
        """ Adds block of (old) convolutional layers to input layer. """
        # 1. Downsample new input layer to old input layer size.
        x = AveragePooling2D()(input_layer)
        # 2. Connect old input processing layers to new input.
        x = old_model.layers[1](x)
        return x

    def _add_new_init_conv_block(self, input_layer, stage):
        """ Adds block of (new) convolutional layers to input layer. """
        # 1. Add (1x1) convolutional layer.
        x = Conv2D(filters=self._number_of_filters(stage), kernel_size=(1, 1), padding='same', activation=LeakyReLU(0.20),
                   kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const)(input_layer)
        # 2. Add first (3x3) convolutional layer.
        x = Conv2D(filters=self._number_of_filters(stage), kernel_size=(3, 3), padding='same', activation=LeakyReLU(0.20),
                   kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const)(x)
        # 3. Add second (3x3) convolutional layer.
        x = Conv2D(filters=self._number_of_filters(stage-1), kernel_size=(3, 3), padding='same', activation=LeakyReLU(0.20),
                   kernel_initializer=self.kernel_init, kernel_constraint=self.kernel_const)(x)
        # 3. Add pooling layer to reduce image resolution.
        x = AveragePooling2D()(x)
        return x

    @staticmethod
    def _double_input_res(old_model):
        """ Doubles the resolution of the input layer. """
        new_res = (
            old_model.input.shape[1] * 2,  # Double width.
            old_model.input.shape[2] * 2,  # Double height.
            old_model.input.shape[3],  # Same channels.
        )
        return Input(new_res)

    @staticmethod
    def _compile_model(model):
        """ Compiles a model using default settings. """
        model.compile(
            loss=wasserstein_loss_drift_penalty,
            optimizer=Adam(
                lr=0.001,
                beta_1=0.00,
                beta_2=0.99,
                epsilon=10e-8
            )
        )

    def _number_of_filters(self, stage):
        return int(np.minimum(self.base_filters / (2**(stage+1)), self.max_filters))
