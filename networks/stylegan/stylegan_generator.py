import tensorflow as tf
from tensorflow.keras.models import Model
from networks.loss import generator_loss


class StyleGANGenerator(Model):

    def __init__(self, *args, **kwargs):
        super(StyleGANGenerator, self).__init__(*args, **kwargs)
        self.loss_type = None
        self.latent_dist = None
        self.n_grad_acc_steps = 16

    def compile(self, optimizer):
        super().compile(optimizer=optimizer)

    def train_on_batch(self, z_latent, discriminator, batch_size, image_augmenter):
        # 0. Initialize accumulated gradient list.
        accum_gradients = [tf.zeros_like(var, dtype='float32') for var in self.variables]
        for _ in range(self.n_grad_acc_steps):
            # 1. Construct loss dict.
            with tf.GradientTape() as tape:
                loss_dict = generator_loss(discriminator, self, image_augmenter, z_latent, self.loss_type)
            # 2. Compute gradients.
            gradients = tape.gradient(loss_dict['g_loss_total'], self.variables)
            # 3. Update accumulated gradients.
            accum_gradients = [(acum_grad + (grad / self.n_grad_acc_steps)) for acum_grad, grad in zip(accum_gradients, gradients)]
        # 3. Apply gradients.
        self.optimizer.apply_gradients(zip(accum_gradients, self.variables))
        return loss_dict
