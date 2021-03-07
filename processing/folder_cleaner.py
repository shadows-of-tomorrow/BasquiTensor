import os
import cv2


class FolderCleaner:

    def __init__(self, dir_in, dir_out):
        self.dir_in = dir_in
        self.dir_out = dir_out
        self.shape_out = (256, 256)

    def clean_folder(self):
        names = os.listdir(self.dir_in)
        for k in range(len(names)):
            name_out = f"{k}.png"
            img = self._read_image(names[k])
            if img is not None:
                img = self._resize_image(img)
                self._write_image(img, name_out)

    def _read_image(self, name):
        img = cv2.imread(os.path.join(self.dir_in, name))
        return img

    def _write_image(self, img, name):
        cv2.imwrite(os.path.join(self.dir_out, name), img)

    def _resize_image(self, img):
        return cv2.resize(img, self.shape_out)


if __name__ == "__main__":
    dir_in = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'io', 'output', 'scraping')
    dir_out = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'io', 'input', 'images', 'bob_ross_fan')
    cleaner = FolderCleaner(dir_in, dir_out)
    cleaner.clean_folder()

