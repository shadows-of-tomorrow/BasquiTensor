import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from datetime import datetime
from networks.network_factory import NetworkFactory
from training.network_trainer import NetworkTrainer
from processing.image_processor import ImageProcessor
from processing.config_processor import ConfigProcessor


class TrainingEngine:

    def __init__(self, run_id=datetime.now().strftime("%Y%m%dT%H%M%S")):
        self.run_id = run_id
        self.parent_dir = os.path.dirname(os.path.dirname(__file__))
        self.config_dir = os.path.join(self.parent_dir, 'io', 'input', 'configs')

    def run(self):
        configs = self._process_config_files()
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        with strategy.scope():
            image_processors = self._construct_image_processors(configs)
            network_trainers = self._construct_network_trainers(image_processors, configs)
            networks = self._construct_networks(configs)
            self._train_networks(networks, network_trainers)

    def _process_config_files(self):
        configs = ConfigProcessor().run(self.parent_dir, self.config_dir)
        return configs

    def _construct_image_processors(self, configs):
        input_dirs = [os.path.join(self.parent_dir, config['directories']['input']) for config in configs]
        output_dirs = [os.path.join(self.parent_dir, config['directories']['output']) for config in configs]
        image_processors = [ImageProcessor(dir_in=input_dirs[k], dir_out=output_dirs[k]) for k in range(len(configs))]
        return image_processors

    @staticmethod
    def _construct_networks(configs):
        print("Constructing networks...")
        network_configs = [config['network_parameters'] for config in configs]
        network_constructors = [NetworkFactory(**network_config) for network_config in network_configs]
        networks_list = [network_constructor.run() for network_constructor in network_constructors]
        return networks_list

    @staticmethod
    def _construct_network_trainers(image_processors, configs):
        training_configs = [config['training_parameters'] for config in configs]
        network_trainers = [NetworkTrainer(image_processors[k], **training_configs[k]) for k in range(len(configs))]
        return network_trainers

    @staticmethod
    def _train_networks(networks, network_trainers):
        assert len(networks) == len(network_trainers)
        for k in range(len(network_trainers)):
            network_trainers[k].run(networks[k])
