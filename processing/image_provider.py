import os
import cv2
import numpy as np
from PIL import Image


class ImageProvider:
    """ This class handles the reading and processing of images. """
    def __init__(self,
                 dir_in: str,
                 x_range=(0, 255),
                 y_range=(-1, 1)):
        # Assign fields to class.
        self.dir_in = dir_in
        self.file_names = os.listdir(dir_in)
        self.x_range = x_range
        self.y_range = y_range
        self.a, self.b = self.calc_shift_coeff(self.x_range, self.y_range)

    def sample_batch(self, batch_size):
        batch = []
        batch_file_names = np.random.choice(self.file_names, batch_size)
        for batch_file_name in batch_file_names:
            img_dir = os.path.join(self.dir_in, batch_file_name)
            arr = self.read_img(img_dir)
            arr = self.shift_arr(arr)
            batch.append(arr)
        return np.stack(batch, axis=0)

    @staticmethod
    def scale_imgs(imgs, shape):
        imgs_out = []
        for img in imgs:
            img_new = cv2.resize(src=img, dsize=shape, interpolation=cv2.INTER_NEAREST)
            imgs_out.append(img_new)
        return imgs_out

    def count_imgs(self):
        return len(os.listdir(self.dir_in))

    @staticmethod
    def read_img(dir_img):
        img = Image.open(dir_img)
        return np.asarray(img)

    def shift_arr(self, arr):
        """ Shifts an array from [x1, x2] to [y1, y2]. """
        return (arr - self.a) / self.b

    @staticmethod
    def calc_shift_coeff(x, y):
        """ Computes the coefficients used to shift an array. """
        a = (x[1] * (y[0] / y[1]) - x[0]) / (y[0] / y[1] - 1.0)  # Location
        b = (x[0] - x[1]) / (y[0] - y[1])  # Scale
        return a, b
