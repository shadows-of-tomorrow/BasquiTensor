import os
import numpy as np
import tensorflow as tf
from PIL import Image


class ImageProcessor:
    """
    Handles the reading and processing of image tensors.
    """
    def __init__(self, dir_in=None, dir_out=None, old_range=(0, 255), new_range=(-1, 1)):
        self.dir_in = dir_in
        self.dir_out = dir_out
        self.n_images = len(os.listdir(self.dir_in))
        self.file_names = os.listdir(self.dir_in)
        self.old_range = old_range
        self.new_range = new_range
        self.a, self.b = self.compute_shift_coefficients(self.old_range, self.new_range)

    def sample_tensor(self, n_samples):
        numpy_arrays = []
        sampled_file_names = np.random.choice(self.file_names, n_samples)
        for sample_file_name in sampled_file_names:
            sample_dir = os.path.join(self.dir_in, sample_file_name)
            numpy_array = self.load_image(sample_dir)
            numpy_arrays.append(numpy_array)
        stacked_array = np.stack(numpy_arrays, axis=0)
        return tf.convert_to_tensor(value=stacked_array, dtype='float32')

    def transform_tensor(self, tensor, transform_type):
        if transform_type == "old_to_new":
            return self.shift_tensor(tensor, self.a, self.b)
        elif transform_type == "old_to_zero_one":
            return self.shift_tensor(tensor, self.old_range[0], self.old_range[1] - self.old_range[0])
        elif transform_type == "min_max_to_zero_one":
            return self._channel_wise_min_max(tensor)
        elif transform_type == "min_max_to_zero_eager":
            return self._channel_wise_min_max_eager(tensor)
        elif transform_type == "new_to_zero_one":
            a, b = self.compute_shift_coefficients(self.new_range, [0, 1])
            return self.shift_tensor(tensor, a, b)
        else:
            return tensor

    @staticmethod
    def _channel_wise_min_max_eager(x):
        x = x.numpy()
        for k in range(x.shape[0]):
            for l in range(x[k].shape[2]):
                x_channel = x[k, :, :, l]
                x_min = tf.reduce_min(x_channel)
                x_max = tf.reduce_max(x_channel)
                x[k, :, :, l] = (x_channel - x_min) / (x_max - x_min)
        x = tf.convert_to_tensor(x)
        return x

    @staticmethod
    def _channel_wise_min_max(x):
        x = tf.Variable(x)
        for k in range(x.shape[0]):
            for l in range(x[k].shape[2]):
                x_channel = x[k, :, :, l]
                x_min = tf.reduce_min(x_channel)
                x_max = tf.reduce_max(x_channel)
                x[k, :, :, l].assign((x_channel - x_min) / (x_max - x_min))
        x = tf.convert_to_tensor(x)
        return x

    @staticmethod
    def resize_tensor(tensor, shape):  # (n_samples, width, height, n_channels)
        return tf.image.resize(images=tensor, size=shape, method=tf.image.ResizeMethod.BILINEAR)

    @staticmethod
    def shift_tensor(numpy_array, a, b):
        return (numpy_array - a) / b

    @staticmethod
    def inverse_shift_tensor(numpy_array, a, b):
        return numpy_array * b + a

    @staticmethod
    def load_image(dir_img):
        img = Image.open(dir_img)
        return np.asarray(img)

    @staticmethod
    def compute_shift_coefficients(old_range, new_range):
        a = (old_range[1] * (new_range[0] / new_range[1]) - old_range[0]) / (new_range[0] / new_range[1] - 1.0)
        b = (old_range[0] - old_range[1]) / (new_range[0] - new_range[1])
        return a, b