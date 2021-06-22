import os
import cv2
import numpy as np
from PIL import Image


class DuplicateCleaner:

    def __init__(self, input_folder, output_folder, hash_size=8):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.hash_size = hash_size
        self.n_dirty_images = len(os.listdir(self.input_folder))
        self.duplicate_counter = 0

    def run(self):
        self._remove_duplicates()

    def _remove_duplicates(self):
        hash_dict = self._construct_hash_dict()
        self._transfer_unique_imgs(hash_dict)

    def _transfer_unique_imgs(self, hash_dict):
        for key, value in hash_dict.items():
            old_path = value[0]
            new_path = f"{self.output_folder}/{key}.png"
            self._transfer_image(old_path, new_path)

    def _transfer_image(self, old_path, new_path):
        with open(old_path, 'rb') as f:
            img = f.read()
        with open(new_path, "wb") as f:
            f.write(img)

    def _construct_hash_dict(self):
        hashes = {}
        for img_name in os.listdir(self.input_folder):
            self._update_hashes(img_name, hashes)
        return hashes

    def _update_hashes(self, img_name, hashes):
        try:
            img_path = f"{self.input_folder}/{img_name}"
            image = np.asarray(Image.open(img_path).convert('RGB'))
            img_hash = self._hash_image(image)
            paths = hashes.get(img_hash, [])
            paths.append(img_path)
            hashes[img_hash] = paths
            self.duplicate_counter += 1
            self._print_hash_process()
        except Exception as e:
            print(f"Exception: {e}")
            self.duplicate_counter += 1

    def _print_hash_process(self):
        pct_complete = (self.duplicate_counter / self.n_dirty_images)*100
        if self.duplicate_counter % 100 == 0:
            message = f"Hash Progress: {pct_complete:.2f}%"
            print(message)

    def _hash_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.hash_size+1, self.hash_size))
        grad = image[:, 1:] > image[:, :-1]
        img_hash = sum([2 ** i for (i, v) in enumerate(grad.flatten()) if v])
        return img_hash
