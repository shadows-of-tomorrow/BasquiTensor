import os
import cv2.cv2 as cv2
import threading


class ResolutionMapper:
    """
    Maps all images in several folders to a single output resolution.
    """
    def __init__(self, input_paths, output_path, target_shape):
        self.input_paths = input_paths
        self.output_path = output_path
        self.target_shape = target_shape

    def run(self, n_threads):
        image_paths = self._get_image_paths()
        image_paths_chunked = self._chunk_image_paths(image_paths, n_threads)
        for chunk in image_paths_chunked:
            thread = threading.Thread(target=lambda: self._process_chunk(chunk))
            thread.start()

    def _process_chunk(self, chunk):
        for image_path in chunk:
            image = self._read_image(image_path)
            image = self._resize_image(image, self.target_shape)
            image_name = self._get_file_name(image_path)
            self._store_image(image, image_name)

    def _store_image(self, image, image_name):
        cv2.imwrite(f"{self.output_path}{image_name}", image)

    def _get_file_name(self, image_path):
        return image_path.split('/')[-1]

    def _resize_image(self, image, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        return image

    def _read_image(self, image_path):
        return cv2.imread(image_path)

    def _get_image_paths(self):
        image_paths = []
        for input_path in self.input_paths:
            image_paths += [f"{input_path}{file_name}" for file_name in os.listdir(input_path)]
        return image_paths

    @staticmethod
    def _chunk_image_paths(image_paths, n_chunks):
        image_paths_chunked = []
        n_image_paths = len(image_paths)
        chunk_size = n_image_paths // n_chunks
        for k in range(n_chunks):
            chunk = image_paths[k * chunk_size:(k + 1) * chunk_size]
            image_paths_chunked.append(chunk)
        if n_image_paths % n_chunks != 0:
            image_paths_chunked[-1] += image_paths[n_chunks * chunk_size:]
        return image_paths_chunked
