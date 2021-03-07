import tensorflow as tf
from tensorflow.keras.models import Model
from networks.loss import generator_loss


class StyleGANGenerator(Model):

    def __init__(self, *args, **kwargs):
        super(StyleGANGenerator, self).__init__(*args, **kwargs)
        self.loss_type = "wasserstein"
        self.latent_dist = "gaussian"
        self.n_grad_acc_steps = 1

    def compile(self, optimizer):
        super().compile(optimizer=optimizer)

    def train_on_batch(self, z_latent, discriminator, batch_size, image_augmenter):
        # 1. Construct loss dict.
        with tf.GradientTape() as tape:
            loss_dict = generator_loss(discriminator, self, image_augmenter, z_latent, self.loss_type)
        # 2. Compute gradients.
        gradients = tape.gradient(loss_dict['g_loss_total'], self.variables)
        # 3. Apply gradients.
        self.optimizer.apply_gradients(zip(gradients, self.variables))
        return loss_dict
