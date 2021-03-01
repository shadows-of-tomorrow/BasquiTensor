import os
import unittest
from networks.checkpoint import load_network_checkpoint
from processing.image_processor import ImageProcessor
from training.network_trainer import NetworkTrainer


class TestTrainingCheckpoint(unittest.TestCase):

    def test_main(self):
        network_path = os.path.join(os.path.dirname(__file__), 'io', 'input', 'networks')
        networks = load_network_checkpoint(network_path)
        network_trainer = self._construct_network_trainer()
        network_trainer.run(networks)

    def _construct_network_trainer(self):
        image_processor = self._construct_image_processor()
        training_config = self._construct_training_config()
        return NetworkTrainer(image_processor, **training_config)

    def _construct_training_config(self):
        return {'monitor_fid': 'False', 'n_batches': [1], 'n_images': [1]}

    def _construct_image_processor(self):
        dir_in = os.path.join(os.path.dirname(__file__), 'io', 'input', 'images')
        dir_out = os.path.join(os.path.dirname(__file__), 'io', 'output', 'training', 'checkpoint')
        return ImageProcessor(dir_in=dir_in, dir_out=dir_out)


if __name__ == "__main__":
    unittest.main()
