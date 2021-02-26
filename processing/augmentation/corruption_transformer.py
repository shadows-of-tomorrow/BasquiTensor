import tensorflow as tf
from processing.augmentation.matrix_construction import apply_probability_mask


class CorruptionTransformer:

    def __init__(self, p_augment):
        self.p_augment = p_augment
        # Image-space corruption probabilities.
        self.p_noise = 1.00
        self.p_cutout = 1.00
        # Image-space corruption intensities.
        self.noise_std = 0.1
        self.cutout_size = 0.5

    def transform_tensors(self, x, batch_size, width, height, channels):
        return self._apply_corruption_transforms(x, batch_size, width, height, channels)

    def _apply_corruption_transforms(self, x, batch_size, width, height, channels):
        x = self._add_rgb_noise(x, batch_size, width, height, channels)
        x = self._apply_cutout(x, batch_size, width, height)
        return x

    def _add_rgb_noise(self, x, batch_size, width, height, channels):
        noise_sigma = tf.abs(tf.random.normal([batch_size], 0, self.noise_std))
        noise_sigma = apply_probability_mask(self.p_augment * self.p_noise, noise_sigma, 0)
        noise_sigma = tf.reshape(noise_sigma, [-1, 1, 1, 1])
        c_noise_sigma = tf.random.normal([batch_size, width, height, channels]) * noise_sigma
        return x + c_noise_sigma

    def _apply_cutout(self, x, batch_size, width, height):
        size = tf.fill([batch_size, 2], self.cutout_size)
        size = apply_probability_mask(self.p_augment * self.p_cutout, size, 0)
        center = tf.random.uniform([batch_size, 2], 0, 1)
        size = tf.reshape(size, [batch_size, 2, 1, 1, 1])
        center = tf.reshape(center, [batch_size, 2, 1, 1, 1])
        coord_x = tf.reshape(tf.range(width, dtype=tf.float32), [1, width, 1, 1])
        coord_y = tf.reshape(tf.range(height, dtype=tf.float32), [1, 1, height, 1])
        mask_x = (tf.abs((coord_x + 0.5) / width - center[:, 0]) >= size[:, 0] / 2)
        mask_y = (tf.abs((coord_y + 0.5) / height - center[:, 1]) >= size[:, 1] / 2)
        c_cutout_mask = tf.cast(tf.logical_or(mask_x, mask_y), tf.float32)
        return x * c_cutout_mask
