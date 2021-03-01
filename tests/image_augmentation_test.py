import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import unittest
import tensorflow as tf
import matplotlib.pyplot as plt
from networks.utils import generate_real_images
from processing.image_processor import ImageProcessor
from processing.augmentation.image_augmenter import ImageAugmenter


class TestImageAugmenter(unittest.TestCase):

    def test_manual_image_inspection(self):
        # 1. Sample images from folder.
        n_samples = 2
        dir_in = os.path.join(os.path.dirname(__file__), 'io', 'input', 'images')
        image_processor = ImageProcessor(dir_in=dir_in)
        x_org = self._sample_images(image_processor, n_samples)
        # 2. Augment sampled images.
        augmenter = self._construct_image_augmenter(p_augment=0.20)
        x_aug = augmenter.augment_tensors(x_org)
        # 2. Construct and store image grid.
        x_org = image_processor.transform_numpy_array(x_org.numpy(), transform_type="min_max_to_zero_one")
        x_aug = image_processor.transform_numpy_array(x_aug.numpy(), transform_type="min_max_to_zero_one")
        plt.suptitle("Original / Transformed")
        for k in range(n_samples):
            plt.subplot(n_samples, 2, 2*k+1)
            plt.imshow(x_org[k])
            plt.subplot(n_samples, 2, 2*(k+1))
            plt.imshow(x_aug[k])
        plt.savefig('inspection_img.png')
        plt.close()
        # 3. Ask tester to inspect images.
        manual_inspection = input("Are the images transformed properly?")
        os.remove('inspection_img.png')
        assert manual_inspection.lower() == "yes"

    def _sample_images(self, img_processor, n_samples):
        x = generate_real_images(img_processor, n_samples, shape=(256, 256), transform_type="old_to_new")
        return tf.convert_to_tensor(x)

    def _construct_image_augmenter(self, p_augment):
        augmenter = ImageAugmenter(p_augment=p_augment)
        return augmenter


if __name__ == '__main__':
    unittest.main()
