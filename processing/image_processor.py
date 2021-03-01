import os
import cv2
import numpy as np
from PIL import Image
from numba import jit


class ImageProcessor:
    """ Handles the reading and processing of images. """
    def __init__(self, dir_in=None, dir_out=None, old_range=(0, 255), new_range=(-1, 1)):
        self.dir_in = dir_in
        self.dir_out = dir_out
        self.n_images = len(os.listdir(self.dir_in))
        self.file_names = os.listdir(self.dir_in)
        self.old_range = old_range
        self.new_range = new_range
        self.a, self.b = self.compute_shift_coefficients(self.old_range, self.new_range)

    def sample_numpy_array(self, n_samples):
        numpy_arrays = []
        sampled_file_names = np.random.choice(self.file_names, n_samples)
        for sample_file_name in sampled_file_names:
            sample_dir = os.path.join(self.dir_in, sample_file_name)
            numpy_array = self.read_img(sample_dir)
            numpy_arrays.append(numpy_array)
        return np.stack(numpy_arrays, axis=0)

    def transform_numpy_array(self, numpy_array, transform_type):
        if transform_type == "old_to_new":
            return self.shift_numpy_array(numpy_array, self.a, self.b)
        elif transform_type == "old_to_zero_one":
            return self.shift_numpy_array(numpy_array, self.old_range[0], self.old_range[1]-self.old_range[0])
        elif transform_type == "min_max_to_zero_one":
            return self._channel_wise_min_max(numpy_array)
        elif transform_type == "new_to_zero_one":
            a, b = self.compute_shift_coefficients(self.new_range, [0, 1])
            return self.shift_numpy_array(numpy_array, a, b)
        else:
            return numpy_array

    @staticmethod
    @jit(nopython=True)
    def _channel_wise_min_max(x):
        for k in range(x.shape[0]):
            for l in range(x[k].shape[2]):
                x_channel = x[k, :, :, l]
                x[k, :, :, l] = (x_channel - x_channel.min()) / (x_channel.max() - x_channel.min())
        return x

    @staticmethod
    def resize_numpy_array(numpy_array, shape):  # (n_samples, width, height, n_channels)
        resized_arrays = []
        for k in range(numpy_array.shape[0]):
            temp_array = cv2.resize(src=numpy_array[k], dsize=shape, interpolation=cv2.INTER_LINEAR)
            resized_arrays.append(temp_array)
        return np.stack(resized_arrays, axis=0)

    @staticmethod
    def read_img(dir_img):
        img = Image.open(dir_img)
        return np.asarray(img)

    @staticmethod
    def resize_images(images, shape):
        imgs_out = []
        for image in images:
            img_new = cv2.resize(src=image, dsize=shape, interpolation=cv2.INTER_NEAREST)
            imgs_out.append(img_new)
        return imgs_out

    @staticmethod
    @jit(nopython=True)
    def shift_numpy_array(numpy_array, a, b):
        return (numpy_array - a) / b

    @staticmethod
    @jit(nopython=True)
    def inverse_shift_numpy_array(numpy_array, a, b):
        return numpy_array * b + a

    @staticmethod
    def compute_shift_coefficients(old_range, new_range):
        a = (old_range[1] * (new_range[0] / new_range[1]) - old_range[0]) / (new_range[0] / new_range[1] - 1.0)
        b = (old_range[0] - old_range[1]) / (new_range[0] - new_range[1])
        return a, b