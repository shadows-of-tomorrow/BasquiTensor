from keras import Model, initializers
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Flatten, Dense


class Discriminator:
    """ This network has to detect whether a painting is legit. """

    def __init__(self, network_architecture: dict):
        self.input_dim = network_architecture['input_dim']
        self.conv_layers = network_architecture['layers']
        self.conv_filters = network_architecture['conv_filters']
        self.conv_kernel_size = network_architecture['conv_kernel_size']
        self.conv_strides = network_architecture['onv_strides']
        self.batch_norm_momentum = network_architecture['batch_norm_momentum']
        self.activation_function = network_architecture['activation_function']
        self.dropout_rate = network_architecture['dropout_rate']
        self.learning_rate = network_architecture['learning_rate']

    def _build_discriminator(self):
        """ Builds the discriminator network of the GAN. """
        discriminator_input = self._generate_input_layer()
        x = discriminator_input
        x = self._add_convolutional_layers(x)
        x = self._add_output_layer(x)
        discriminator_output = x
        return Model(discriminator_input, discriminator_output)

    def _generate_input_layer(self):
        """ Creates an input layer. """
        return Input(shape=self.input_dim, name='discriminator_input')

    def _add_convolutional_layers(self, x):
        """ Adds convolutional layers to a network. """
        for i in range(self.conv_layers):

            # Add convolutional layer.
            x = Conv2D(
                filters=self.conv_filters[i],
                kernel_size=self.conv_kernel_size[i],
                strides=self.conv_strides[i],
                padding='same',
                name='discriminator_conv_' + str(i)
            )

            # Add batch normalization to layer.
            if self.batch_norm_momentum and i > 0:
                x = BatchNormalization(momentum=self.batch_norm_momentum)(x)

            # Add activation function to layer.
            x = Activation(self.activation_function)(x)

            # Add dropout to layer.
            if self.dropout_rate:
                x = Dropout(rate=self.dropout_rate)(x)

        return x

    def _add_output_layer(self, x):
        """ Adds an output layer to a network. """
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid', kernel_initializer=initializers.RandomNormal(stddev=0.01))(x)
        return x
