import os
from PIL import Image, ImageOps


class ImageProcessor:
    """ Transforms all images in a directory to a uniform format. """
    def __init__(self, dir_in, dir_out, dims):
        self.dir_in = dir_in
        self.dir_out = dir_out
        self.dims = dims

    def execute(self):
        ids = self._image_ids()
        for id in ids:
            img_in = Image.open(self.dir_in / id)
            img_out = self._resize_image(img_in)
            img_out.save(self.dir_out / id)

    def _pad_image(self, image):
        img_size = image.size
        delta_w = self.dims[0] - img_size[0]
        delta_h = self.dims[1] - img_size[1]
        padding = (delta_w//2, delta_h//2, delta_w//2, delta_h//2)
        return ImageOps.expand(image, padding)

    def _resize_image(self, image):
        return image.resize(self.dims)

    def _max_dims(self):
        ids = self._image_ids()
        max_dims = [0, 0]
        for id in ids:
            img = Image.open(self.dir_in / id)
            size = img.size
            if size[0] > max_dims[0]:
                max_dims[0] = size[0]
            if size[1] > max_dims[1]:
                max_dims[1] = size[1]
        return tuple(max_dims)

    def _image_ids(self):
        return os.listdir(self.dir_in)
