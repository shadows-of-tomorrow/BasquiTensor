import tensorflow as tf
from tensorflow.keras.models import Model
from networks.utils.sampling import generate_real_images, generate_fake_images
from networks.utils.custom_loss import discriminator_loss


class StyleGANDiscriminator(Model):

    def __init__(self, *args, **kwargs):
        super(StyleGANDiscriminator, self).__init__(*args, **kwargs)
        self.step_tracker = 1
        self.loss_type = "wasserstein"
        self.ada_target = 0.00
        self.ada_smoothing = 0.999

    def compile(self, **kwargs):
        super().compile(**kwargs)

    def train_on_batch(self, image_processor, generator, batch_size, shape, image_augmenter):
        loss_dict = self._distributed_train_step(image_processor, generator, batch_size, shape, image_augmenter)
        return loss_dict

    def _distributed_train_step(self, image_processor, generator, batch_size, shape, image_augmenter):
        loss_dict_replicas = self.distribute_strategy.run(self._train_step, args=(image_processor, generator, batch_size, shape, image_augmenter))
        loss_dict = self.distribute_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_dict_replicas, axis=None)
        aug_dict = self._adjust_augmentation_probability(image_processor, image_augmenter, batch_size, shape)
        return {**loss_dict, **aug_dict}

    @tf.function
    def _train_step(self, image_processor, generator, batch_size, shape, image_augmenter):
        # 1. Generate (augmented) real samples.
        x_real = generate_real_images(image_processor, batch_size, shape, transform_type='old_to_new')
        x_real = image_augmenter.augment_tensors(x_real, is_tensor=True)
        # 2. Generate (augmented) fake samples.
        x_fake = generate_fake_images(image_processor, generator, batch_size, shape, transform_type=None)
        x_fake = image_augmenter.augment_tensors(x_fake, is_tensor=True)
        # 3. Construct loss dict.
        with tf.GradientTape() as tape:
            loss_dict, y_real = discriminator_loss(self, x_real, x_fake, batch_size, self.loss_type)
        # 4. Compute gradients.
        gradients = tape.gradient(loss_dict['d_loss_total'], self.variables)
        # 5. Apply gradients.
        self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.variables) if grad is not None)
        return loss_dict

    def _adjust_augmentation_probability(self, image_processor, image_augmenter, batch_size, shape):
        x_real = generate_real_images(image_processor, batch_size, shape, transform_type='old_to_new')
        x_real = image_augmenter.augment_tensors(x_real, is_tensor=True)
        y_real = self(x_real, training=False)
        rt = tf.reduce_mean(tf.sign(y_real))
        self.ada_target = self.ada_smoothing * self.ada_target + (1.0-self.ada_smoothing) * rt
        if self.step_tracker % 4 == 0:
            image_augmenter.update_augmentation_probabilities(self.ada_target, batch_size * self.step_tracker)
            self.step_tracker = 1
        else:
            self.step_tracker += 1
        return {"d_ada_target": self.ada_target, "p_augment": image_augmenter.p_augment}

