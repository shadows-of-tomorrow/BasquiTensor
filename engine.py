import os
from datetime import datetime
from processing.configs import ConfigProcessor
from processing.images import ImageProcessor
from construction.networks import NetworkConstructor
from training.networks import NetworkTrainer


class Engine:

    def __init__(self, run_id):
        self.run_id = run_id
        self.parent_dir = os.path.dirname(__file__)
        self.config_dir = os.path.join(self.parent_dir, 'configs')

    def run(self):
        configs = self._read_config_files()
        image_processors = self._construct_image_processors(configs)
        networks = self._construct_networks(configs)
        network_trainers = self._construct_network_trainers(image_processors, configs)
        self._train_networks(networks, network_trainers)

    def _read_config_files(self):
        return ConfigProcessor().read_configs(self.config_dir)

    def _construct_image_processors(self, configs):
        input_dirs = [os.path.join(self.parent_dir, config['image_input_folder']) for config in configs]
        output_dirs = [os.path.join(self.parent_dir, config['image_output_folder']) for config in configs]
        image_processors = [ImageProcessor(dir_in=input_dirs[k], dir_out=output_dirs[k]) for k in range(len(configs))]
        return image_processors

    @staticmethod
    def _construct_networks(configs):
        network_configs = [config['network_parameters'] for config in configs]
        network_constructors = [NetworkConstructor(**network_config) for network_config in network_configs]
        networks_list = [network_constructor.execute() for network_constructor in network_constructors]
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
            network_trainers[k].execute(networks[k])


if __name__ == "__main__":
    time_stamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    engine = Engine(run_id=time_stamp)
    engine.run()
