import tensorflow as tf
from tensorflow.keras.models import Model
from networks.utils import generate_fake_images
from networks.utils import generate_real_images
from networks.loss import discriminator_loss


class Discriminator(Model):

    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.step_tracker = 1
        self.loss_type = "vanilla"
        self.ada_target = 0.00
        self.ada_smoothing = 0.999

    def compile(self, optimizer):
        super().compile(optimizer=optimizer)

    def train_on_batch(self, image_processor, generator, batch_size, shape, image_augmenter):
        # 1. Generate (augmented) real samples.
        x_real = generate_real_images(image_processor, batch_size, shape, transform_type='old_to_new')
        x_real = image_augmenter.run(x_real, is_tensor=False)
        # 2. Generate (augmented) fake samples.
        x_fake = generate_fake_images(image_processor, generator, batch_size, shape, transform_type=None)
        x_fake = image_augmenter.run(x_fake, is_tensor=False)
        # 3. Construct loss dict.
        with tf.GradientTape() as tape:
            loss_dict, y_real = discriminator_loss(self, x_real, x_fake, batch_size, self.loss_type)
        # 4. Compute gradients.
        gradient = tape.gradient(loss_dict['d_loss_total'], self.variables)
        # 5. Apply gradients.
        self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradient, self.variables) if grad is not None)
        # 6. Adjust augmentation probability.
        aug_dict = self._adjust_augmentation_probability(y_real, image_augmenter, batch_size)
        return {**loss_dict, **aug_dict}

    def _adjust_augmentation_probability(self, y_real, image_augmenter, batch_size):
        rt = tf.reduce_mean(tf.sign(y_real)).numpy()
        self.ada_target = self.ada_smoothing * self.ada_target + (1.0-self.ada_smoothing) * rt
        if self.step_tracker % 4 == 0:
            image_augmenter.adapt_augmentation_probability(self.ada_target, batch_size * self.step_tracker)
            self.step_tracker = 1
        else:
            self.step_tracker += 1
        return {"d_ada_target": self.ada_target, "p_augment": image_augmenter.p_augment}

