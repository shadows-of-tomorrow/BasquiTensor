import os
from datetime import datetime
from processing.config_processor import ConfigProcessor
from processing.image_processor import ImageProcessor
from construction.network_constructor import NetworkConstructor
from training.network_trainer import NetworkTrainer


class Engine:

    def __init__(self, run_id):
        self.run_id = run_id
        self.parent_dir = os.path.dirname(__file__)
        self.config_dir = os.path.join(self.parent_dir, 'io', 'input', 'configs')

    def run(self):
        configs = self._process_config_files()
        image_processors = self._construct_image_processors(configs)
        networks = self._construct_networks(configs)
        network_trainers = self._construct_network_trainers(image_processors, configs)
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
        network_configs = [config['network_parameters'] for config in configs]
        network_constructors = [NetworkConstructor(**network_config) for network_config in network_configs]
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


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    engine = Engine(run_id=timestamp)
    engine.run()
