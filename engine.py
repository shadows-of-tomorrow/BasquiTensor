import os
from processing.configs import ConfigProcessor
from processing.images import ImageProcessor
from construction.networks import NetworkConstructor
from training.networks import NetworkTrainer


if __name__ == "__main__":

    # 1. Read configuration files.
    parent_dir = os.path.dirname(__file__)
    config_dir = os.path.join(parent_dir, 'configs')
    configs = ConfigProcessor().read_configs(config_dir)

    # 2. Construct list of image processors for each config.
    image_processors = [ImageProcessor(os.path.join(parent_dir, config['image_input_folder']),
                                       os.path.join(parent_dir, config['image_output_folder'])) for config in configs]

    # 3. Construct list of networks for each config.
    network_constructors = [NetworkConstructor(**config['network_parameters']) for config in configs]
    networks_list = [network_constructor.execute() for network_constructor in network_constructors]

    # 4. Train list of networks for each config.
    network_trainers = [NetworkTrainer(image_processors[k], **configs[k]['training_parameters']) for k in range(len(configs))]
    for k in range(len(network_trainers)):
        network_trainers[k].execute(networks_list[k])
