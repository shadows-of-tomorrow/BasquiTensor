import numpy as np
import tensorflow as tf
from processing.augmentation.matrix_construction import apply_probability_mask
from processing.augmentation.matrix_construction import construct_3d_scale_matrix
from processing.augmentation.matrix_construction import construct_3d_rotation_matrix
from processing.augmentation.matrix_construction import construct_3d_translation_matrix


class ColorTransformer:

    def __init__(self, p_augment):
        self.p_augment = p_augment
        # Color transformation probabilities.
        self.p_brightness = 1.00
        self.p_contrast = 1.00
        self.p_luma_flip = 1.00
        self.p_hue_rotation = 1.00
        self.p_saturation = 1.00
        # Color transformation intensities.
        self.brightness_std = 0.20
        self.contrast_std = 0.50
        self.hue_max = 1.0
        self.saturation_std = 1.0

    def transform_tensors(self, x, batch_size, width, height, channels):
        c = self._construct_color_transforms(batch_size, width, height)
        x = self._apply_color_transforms(c, x, batch_size, width, height, channels)
        return x

    def _apply_color_transforms(self, c, x, batch_size, width, height, channels):
        x = tf.transpose(x, [0, 3, 2, 1])
        x = tf.reshape(x, [batch_size, channels, width * height])
        x = c[:, :3, :3] @ x + c[:, :3, 3:]
        x = tf.reshape(x, [batch_size, channels, width, height])
        x = tf.transpose(x, [0, 3, 2, 1])
        return x

    def _construct_color_transforms(self, batch_size, width, height):
        eye = tf.eye(4, batch_shape=[batch_size])
        c = self._multiply_brightness_translation(eye, batch_size)
        c = self._multiply_contrast_scaling(c, batch_size)
        c = self._multiply_luma_flip(c, eye, batch_size)
        c = self._multiply_hue_rotation(c, batch_size)
        c = self._multiply_saturation_scaling(c, eye, batch_size)
        return c

    def _multiply_brightness_translation(self, c, batch_size):
        bright = tf.random.normal([batch_size], 0, self.brightness_std)
        bright = apply_probability_mask(self.p_augment * self.p_brightness, bright, 0)
        c_bright = construct_3d_translation_matrix(bright, bright, bright)
        return c_bright @ c

    def _multiply_contrast_scaling(self, c, batch_size):
        contrast_scale = 2 ** tf.random.normal([batch_size], 0, self.contrast_std)
        contrast_scale = apply_probability_mask(self.p_augment * self.p_contrast, contrast_scale, 1)
        c_contrast_scale = construct_3d_scale_matrix(contrast_scale, contrast_scale, contrast_scale)
        return c_contrast_scale @ c

    def _multiply_luma_flip(self, c, eye, batch_size):
        v = np.asarray([1, 1, 1, 0]) * np.sqrt(1.0/3.0)
        v_outer = np.outer(v, v)
        luma_flip = tf.floor(tf.random.uniform([batch_size], 0, 2))
        luma_flip = apply_probability_mask(self.p_augment * self.p_luma_flip, luma_flip, 0)
        luma_flip = tf.reshape(luma_flip, [batch_size, 1, 1])
        c_luma_flip = (eye - 2.0 * v_outer * luma_flip)
        return c_luma_flip @ c

    def _multiply_hue_rotation(self, c, batch_size):
        v = np.asarray([1, 1, 1, 0]) * np.sqrt(1.0/3.0)
        hue_rotation = tf.random.uniform([batch_size], -np.pi * self.hue_max, np.pi * self.hue_max)
        hue_rotation = apply_probability_mask(self.p_augment * self.p_hue_rotation, hue_rotation, 0)
        c_hue_rotation = construct_3d_rotation_matrix(v, hue_rotation)
        return c_hue_rotation @ c

    def _multiply_saturation_scaling(self, c, eye, batch_size):
        v = np.asarray([1, 1, 1, 0]) * np.sqrt(1.0/3.0)
        v_outer = np.outer(v, v)
        sat_scale = 2 ** tf.random.normal([batch_size], 0, self.saturation_std)
        sat_scale = apply_probability_mask(self.p_augment * self.p_saturation, sat_scale, 1)
        sat_scale = tf.reshape(sat_scale, [batch_size, 1, 1])
        c_sat_scale = (v_outer + (eye - v_outer) * sat_scale)
        return c_sat_scale @ c
