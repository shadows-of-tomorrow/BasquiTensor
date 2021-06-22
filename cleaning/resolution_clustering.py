import os
import cv2.cv2 as cv2
import threading
import numpy as np


class ResolutionClusterer:
    """
    Clusters images in a folder based on resolution buckets.
    """
    def __init__(self, input_path, output_path, buckets):
        self.input_path = input_path
        self.output_path = output_path
        self.buckets = buckets

    def run(self, n_threads):
        self._create_bucket_folders()
        image_names = self._get_image_names()
        chunked_image_names = self._chunk_image_names(image_names, n_threads)
        for chunk in chunked_image_names:
            thread = threading.Thread(target=lambda: self._process_image_names(chunk))
            thread.start()

    def _process_image_names(self, image_names):
        for image_name in image_names:
            try:
                image = self._read_image(image_name)
                image = self._crop_image(image)
                res_bucket = self._get_resolution_bucket(image)
                self._write_image(image, image_name, res_bucket)
            except Exception as e:
                print(f"{e}")

    def _chunk_image_names(self, image_names, n_chunks):
        chunked_image_names = []
        n_image_names = len(image_names)
        chunk_size = n_image_names // n_chunks
        for k in range(n_chunks):
            chunk = image_names[k*chunk_size:(k+1)*chunk_size]
            chunked_image_names.append(chunk)
        if n_image_names % n_chunks != 0:
            chunked_image_names[-1] += image_names[n_chunks*chunk_size:]
        return chunked_image_names

    def _write_image(self, image, image_name, res_bucket):
        image_path = f"{self.output_path}{res_bucket}/{image_name}"
        cv2.imwrite(image_path, image)

    def _get_resolution_bucket(self, image):
        width, height = self._get_image_shape(image)
        for key, value in self.buckets.items():
            if value[0] <= np.minimum(width, height) < value[1]:
                return key

    def _read_image(self, image_name):
        return cv2.imread(f"{self.input_path}{image_name}")

    def _get_image_names(self):
        return os.listdir(self.input_path)

    def _get_image_shape(self, image):
        return image.shape[0], image.shape[1]

    def _crop_image(self, image):
        width, height = self._get_image_shape(image)
        if width > height:
            return image[:height, :]
        elif height > width:
            return image[:, :width]
        else:
            return image

    def _create_bucket_folders(self):
        for key in self.buckets:
            bucket_path = f"{self.output_path}{key}"
            if not os.path.exists(bucket_path):
                os.makedirs(bucket_path)