import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.initializers import HeNormal
from networks.style_gan.discriminator import Discriminator
from networks.layers import MinibatchStDev


class DiscriminatorConstructorStyle:

    def __init__(self, **network_config):
        # Resolution related fields.
        self.output_res = network_config['output_res']
        self.output_res_log2 = int(np.log2(self.output_res))
        self.n_blocks = int(self.output_res_log2 - 1)
        # Filter related fields.
        self.n_base_filters = network_config['n_base_filters']
        self.n_max_filters = network_config['n_max_filters']
        # Other fields.
        self.adam_params = network_config['adam_params']
        self.kernel_init = HeNormal()

    def run(self):
        # 1. Construct initial block.
        backend.set_floatx('float32')
        input_layer, x, y = self._construct_initial_block()
        # 2. Add intermediate blocks.
        for stage in range(2, self.n_blocks):
            x, y = self._add_intermediate_block(x, y, stage)
        # 3. Add terminal block.
        output_layer = self._add_terminal_block(x)
        # 4. Construct and compile discriminator.
        discriminator = Discriminator(input_layer, output_layer)
        self._compile_model(discriminator)
        return [[discriminator, discriminator]]

    def _construct_initial_block(self):
        n_filters_1 = self._compute_n_filters_at_stage(1)
        n_filters_2 = self._compute_n_filters_at_stage(2)
        # 1. Construct input layer.
        input_layer = self._construct_input_layer()
        # 2. Map input image (RGB) to network.
        x = self._add_from_rgb_layer(input_layer, n_filters_1)
        # 3.1 Directly apply down sampling layer to previous block.
        y = self._add_downsampling_layer(x, n_filters_2, True)
        # 3.2 Add two (3x3) convolutional layers and down sampling layer.
        x = self._add_convolutional_layers(x, n_filters_1, 3, 1)
        x = self._add_convolutional_layers(x, n_filters_2, 3, 1)
        x = self._add_downsampling_layer(x, None, False)
        # 4. Combine x and y to form block.
        x = (x + y) / np.sqrt(2.0)
        return input_layer, x, y

    def _add_intermediate_block(self, x, y, stage):
        n_filters_1 = self._compute_n_filters_at_stage(stage)
        n_filters_2 = self._compute_n_filters_at_stage(stage + 1)
        # 1. Directly pass y to down sampling layer.
        y = self._add_downsampling_layer(y, n_filters_2, True)
        # 2. Add two convolutional layers and down sampling layer to x.
        x = self._add_convolutional_layers(x, n_filters_1, 3, 1)
        x = self._add_convolutional_layers(x, n_filters_2, 3, 1)
        x = self._add_downsampling_layer(x, None, False)
        # 3. Combine x and y to form block.
        x = (x + y) / np.sqrt(2.0)
        return x, y

    def _add_terminal_block(self, x):
        n_filters = self._compute_n_filters_at_stage(self.n_blocks)
        # 1. Add mini-batch standard deviation layer.
        x = self._add_minibatch_std_layer(x)
        # 2. Add single (3x3) convolutional layer.
        x = self._add_convolutional_layers(x, n_filters, 3, 1)
        # 3. Add intermediate dense layer.
        x = self._add_dense_layer(x, n_filters, True, True)
        # 4. Add output dense layer.
        x = self._add_dense_layer(x, 1, False, False)
        return x

    def _compile_model(self, model):
        lr = self.adam_params["lr"]
        beta_1 = self.adam_params["beta_1"]
        beta_2 = self.adam_params["beta_2"]
        epsilon = self.adam_params["epsilon"]
        model.compile(optimizer=Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon))

    def _construct_input_layer(self):
        return Input(shape=(self.output_res, self.output_res, 3))

    def _add_convolutional_layers(self, x, n_filters, kernel_size, n_layers):
        for _ in range(n_layers):
            x = Conv2D(filters=n_filters,
                       kernel_size=(kernel_size, kernel_size),
                       padding='same',
                       kernel_initializer=self.kernel_init,
                       activation=LeakyReLU(0.20))(x)
        return x

    def _add_dense_layer(self, x, units, flatten, use_activation):
        if flatten:
            x = Flatten()(x)
        x = Dense(units=units, kernel_initializer=self.kernel_init)(x)
        if use_activation:
            x = LeakyReLU(0.20)(x)
        return x

    def _add_from_rgb_layer(self, x, n_filters):
        x = self._add_convolutional_layers(x, n_filters, 1, 1)
        return x

    def _add_downsampling_layer(self, x, n_filters, add_rgb_layer):
        x = AveragePooling2D()(x)
        if add_rgb_layer:
            x = self._add_from_rgb_layer(x, n_filters)
        return x

    def _add_minibatch_std_layer(self, x):
        return MinibatchStDev()(x)

    def _compute_n_filters_at_stage(self, stage):
        return np.minimum(int(self.n_base_filters / (2 ** (self.n_blocks - stage + 1))), self.n_max_filters)
