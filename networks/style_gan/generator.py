import tensorflow as tf
from tensorflow.keras.models import Model


class Generator(Model):

    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)

    def compile(self, optimizer):
        super().compile(optimizer=optimizer)

    def train_on_batch(self, z_latent, discriminator, batch_size, image_augmenter):
        with tf.GradientTape() as tape:
            # 1. Generate fake images with trainable generator.
            x_fake = self(z_latent, training=True)
            # 2. Apply image transformations (learnable gradients).
            x_fake = image_augmenter.run(x_fake, is_tensor=True)
            # 2. Score fake images with non-trainable discriminator.
            y_fake = discriminator(x_fake, training=False)
            # 3. Compute generator loss.
            g_loss = -tf.reduce_mean(y_fake)
        # 4. Compute gradients.
        gradients = tape.gradient(g_loss, self.variables)
        # 5. Apply gradient descent.
        self.optimizer.apply_gradients(zip(gradients, self.variables))
        return {'g_loss': g_loss.numpy()}
