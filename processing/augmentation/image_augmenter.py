import numpy as np
import tensorflow as tf
from processing.augmentation.color_transformer import ColorTransformer
from processing.augmentation.geometry_transformer import GeometryTransformer
from processing.augmentation.corruption_transformer import CorruptionTransformer


class ImageAugmenter:

    def __init__(self):
        # Set augmentation parameters.
        self.p_augment = 0.00
        self.p_augment_target = 0.60
        self.p_augment_threshold = 0.01
        self.p_augment_max = 0.80
        self.n_adjust_imgs = 50000
        # Construct image transformers.
        self.geometry_transformer = GeometryTransformer(self.p_augment)
        self.color_transformer = ColorTransformer(self.p_augment)
        self.corruption_transformer = CorruptionTransformer(self.p_augment)

    def augment_tensors(self, x, is_tensor=True):
        if not is_tensor:
            x = tf.convert_to_tensor(value=x, dtype='float32')
        if self.p_augment > self.p_augment_threshold:
            batch_size, width, height, channels = x.shape.as_list()
            x = self._transform_tensors(x, batch_size, width, height, channels)
        return x

    def update_augmentation_probabilities(self, rt, n_shown_images):
        p_augment_new = self._compute_augmentation_probability(rt, n_shown_images)
        self.p_augment = p_augment_new
        self.geometry_transformer.p_augment = p_augment_new
        self.color_transformer.p_augment = p_augment_new
        self.corruption_transformer = p_augment_new

    def _compute_augmentation_probability(self, rt, n_shown_images):
        p_augment = self.p_augment + (n_shown_images/self.n_adjust_imgs) * np.sign(rt - self.p_augment_target)
        p_augment = np.clip(p_augment, 0.0, self.p_augment_max)
        return p_augment

    def _transform_tensors(self, x, batch_size, width, height, channels):
        x = self.geometry_transformer.transform_tensors(x, batch_size, width, height, channels)
        x = self.color_transformer.transform_tensors(x, batch_size, width, height, channels)
        x = self.corruption_transformer.transform_tensors(x, batch_size, width, height, channels)
        return x
