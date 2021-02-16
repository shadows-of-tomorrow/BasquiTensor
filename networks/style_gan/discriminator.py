import tensorflow as tf
from tensorflow.keras.models import Model
from networks.utils import generate_fake_samples
from networks.utils import generate_real_samples


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
        for k in range(self.d_steps):
            x_fake = generate_fake_samples(
                image_processor=image_processor,
                generator=generator,
                n_samples=batch_size,
                shape=shape,
                transform_type=None
            )
            x_real = generate_real_samples(
                image_processor=image_processor,
                n_samples=batch_size,
                shape=shape,
                transform_type='old_to_new'
            )
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
