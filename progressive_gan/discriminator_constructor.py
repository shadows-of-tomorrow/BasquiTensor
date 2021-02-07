import numpy as np
import tensorflow as tf

from progressive_gan.utils import generate_fake_samples
from progressive_gan.utils import generate_real_samples
from progressive_gan.layers import WeightedSum, MinibatchStDev

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, LeakyReLU, Dense, Conv2D
from tensorflow.keras.layers import AveragePooling2D, Flatten
from tensorflow.keras.initializers import RandomNormal


class DiscriminatorConstructor:
    """ Creates a list of progressively growing discriminator models. """
    def __init__(self, skip_layers=3, **network_config):
        self.input_res = network_config['input_res']
        self.output_res = network_config['output_res']
        self.max_filters = network_config['max_filters']
        self.base_filters = network_config['base_filters']
        self.adam_params = network_config['adam_params']
        self.n_blocks = int(np.log2(self.output_res/self.input_res)+1)
        self.skip_layers = skip_layers
        self.kernel_init = RandomNormal()

    def run(self):
        """ Creates a list of progressively growing discriminator models. """
        # 1. Initialize list of discriminators.
        discriminators = []
        # 2. Construct and add initial discriminator.
        init_models = self._construct_initial_model()
        discriminators.append(init_models)
        # 3. Construct and add next discriminator.
        for k in range(1, self.n_blocks):
            old_model = discriminators[k - 1][0]
            next_models = self._construct_next_models(old_model, k)
            discriminators.append(next_models)
        return discriminators

    def _construct_initial_model(self):
        """ Constructs the initial discriminator. """
        # 1. Compute number of filters for initial model.
        filters_init = self._number_of_filters(0)
        # 2. Construct input layer for initial resolution.
        input_layer = Input(shape=(self.input_res, self.input_res, 3))
        # 3. Add fromRGB layer.
        x = self._add_from_rgb_layer(input_layer, filters_init)
        # 3. Add minibatch standard deviation layer.
        x = MinibatchStDev()(x)
        # 4. Add (3x3) convolutional layer.
        x = Conv2D(filters=filters_init, kernel_size=(3, 3), padding='same', kernel_initializer=self.kernel_init)(x)
        x = LeakyReLU(0.20)(x)
        # 5. Flatten and add dense layer.
        x = Flatten()(x)
        x = Dense(units=filters_init, kernel_initializer=self.kernel_init)(x)
        x = LeakyReLU(0.20)(x)
        # 6. Flatten and add dense layer.
        x = Flatten()(x)
        output_layer = Dense(units=1, kernel_initializer=self.kernel_init)(x)
        # 7. Construct (custom) Keras model.
        initial_model = Discriminator(inputs=input_layer, outputs=output_layer)
        # 8. Compile (custom) Keras model.
        self._compile_model(initial_model)
        return [initial_model, initial_model]

    def _construct_next_models(self, old_model, stage):
        """ Constructs fade-in and new discriminators. """
        # 1. Construct new input layer with double the old resolution.
        input_layer = self._construct_new_input_layer(old_model)
        # 2. Add new block to input layer.
        block_new = self._add_new_block(input_layer, stage)
        # 3. Add fade-in block input layer.
        block_fade_in = self._add_fade_in_block(input_layer, stage)
        # 4. Compute weighted sum of blocks.
        block_wsum = WeightedSum()([block_fade_in, block_new])
        # 5. Add old model layers to both blocks.
        d_next = self._add_old_layers(block_new, old_model)
        d_fade_in = self._add_old_layers(block_wsum, old_model)
        # 6. Construct (custom) Keras models.
        next_model = Discriminator(inputs=input_layer, outputs=d_next)
        fade_in_model = Discriminator(inputs=input_layer, outputs=d_fade_in)
        # 7. Compile (custom) Keras models.
        self._compile_model(next_model)
        self._compile_model(fade_in_model)
        return [next_model, fade_in_model]

    def _add_old_layers(self, new_model, old_model):
        """ Adds the old model layers to the new model (skipping the input and fromRBG layers). """
        for k in range(self.skip_layers, len(old_model.layers)):
            new_model = old_model.layers[k](new_model)
        return new_model

    def _add_fade_in_block(self, input_layer, stage):
        """ Adds block of (old) convolutional layers to input layer. """
        # 1. Add pooling layer to downscale image resolution (0.50x).
        x = AveragePooling2D()(input_layer)
        # 2. Add fromRGB layer.
        x = self._add_from_rgb_layer(x, self._number_of_filters(stage-1))
        return x

    def _add_new_block(self, input_layer, stage):
        """ Adds block of (new) convolutional layers to input layer. """
        # 1. Compute number of filters at the current and previous stage.
        filters_current = self._number_of_filters(stage)
        filters_previous = self._number_of_filters(stage-1)
        # 2. Add fromRGB layer.
        x = self._add_from_rgb_layer(input_layer, filters_current)
        # 2. Add first (3x3) convolutional layer.
        x = Conv2D(filters=filters_current, kernel_size=(3, 3), padding='same', kernel_initializer=self.kernel_init)(x)
        x = LeakyReLU(0.20)(x)
        # 3. Add second (3x3) convolutional layer.
        x = Conv2D(filters=filters_previous, kernel_size=(3, 3), padding='same', kernel_initializer=self.kernel_init)(x)
        x = LeakyReLU(0.20)(x)
        # 3. Add pooling layer to downscale image resolution (0.50x).
        x = AveragePooling2D()(x)
        return x

    def _add_from_rgb_layer(self, x, filters):
        """ Adds a (1x1) convolutional layer to process an RGB image. """
        x = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', kernel_initializer=self.kernel_init)(x)
        x = LeakyReLU(0.20)(x)
        return x

    @staticmethod
    def _construct_new_input_layer(old_model):
        """ Constructs an input layer with twice the resolution of the old model. """
        new_res = (old_model.input.shape[1] * 2, old_model.input.shape[2] * 2, old_model.input.shape[3])
        return Input(new_res)

    def _number_of_filters(self, stage):
        """ Computes the number of feature maps at a given stage. """
        return int(np.minimum(self.base_filters / (2**(stage+1)), self.max_filters))

    def _compile_model(self, model):
        """ Compiles a model using default settings. """
        lr = self.adam_params["lr"]
        beta_1 = self.adam_params["beta_1"]
        beta_2 = self.adam_params["beta_2"]
        epsilon = self.adam_params["epsilon"]
        model.compile(optimizer=Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon))


# Todo: Match signatures of Keras class.
class Discriminator(Model):
    """ Wraps keras model to incorporate gradient penalty. """
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.d_steps = 1
        self.dp_weight = 0.001
        self.gp_weight = 10.0

    def compile(self, optimizer):
        super().compile(optimizer=optimizer)

    def train_on_batch(self, image_processor, generator, batch_size, shape):
        latent_dim = generator.input.shape[1]
        for k in range(self.d_steps):
            x_fake, _ = generate_fake_samples(generator, latent_dim, batch_size)
            x_real, _ = generate_real_samples(image_processor, batch_size, shape)
            with tf.GradientTape() as tape:
                # 1. Compute loss on real examples.
                y_real = self(x_real, training=True)
                d_loss_real = -tf.reduce_mean(y_real)
                # 2. Compute loss on fake examples.
                y_fake = self(x_fake, training=True)
                d_loss_fake = tf.reduce_mean(y_fake)
                # 3. Compute total loss on training examples.
                d_loss = d_loss_real + d_loss_fake
                # 2. Compute gradient penalty.
                gp_loss = self.gp_weight * self._gradient_penalty(x_real, x_fake, batch_size)
                d_loss += gp_loss
                # 3. Compute drift penalty.
                dp_loss = self.dp_weight * tf.reduce_mean(tf.square(y_real))
                d_loss += dp_loss
            # 4. Apply gradients
            gradient = tape.gradient(d_loss, self.variables)
            self.optimizer.apply_gradients(zip(gradient, self.variables))
        # 5. Compile losses in dictionary.
        return {"d_loss_real": d_loss_real.numpy(), "d_loss_fake": d_loss_fake.numpy(), "gp_loss": gp_loss.numpy()}

    def _gradient_penalty(self, x_real, x_fake, batch_size):
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        x_int = (1.0 - alpha) * x_real + alpha * x_fake
        with tf.GradientTape() as tape:
            tape.watch(x_int)
            y_int = self(x_int, training=True)
        gradient = tape.gradient(y_int, [x_int])[0]
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)
        return gradient_penalty