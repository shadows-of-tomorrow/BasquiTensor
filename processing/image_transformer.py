import os
from PIL import Image


class ImageTransformer:
    """ Transforms a collection of images from one format to another. """

    def reshape_dir(self, shape, dir_in, dir_out):
        file_names = os.listdir(dir_in)
        for file_name in file_names:
            img = Image.open(dir_in + "//" + file_name)
            img = self.reshape_img(img, shape)
            img.save(dir_out + "//" + file_name)

    @staticmethod
    def reshape_img(img, new_shape):
        return img.resize(new_shape)
