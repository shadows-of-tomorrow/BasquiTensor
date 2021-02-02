import os
import json


class ConfigProcessor:
    """ Handles the reading and processing of configuration files. """
    def __init__(self, from_json=True):
        self.from_json = from_json

    def run(self, parent_dir, config_dir):
        configs = self.read_configs(config_dir)
        self.store_configs(parent_dir, configs)
        return configs

    def read_configs(self, config_dir):
        if self.from_json is True:
            return self._read_jsons(config_dir)

    @staticmethod
    def store_configs(parent_dir, configs):
        for config in configs:
            file_dir = os.path.join(parent_dir, config['directories']['output'])
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            with open(file_dir + '/config.json', 'w') as file:
                json.dump(config, file)

    @staticmethod
    def _read_jsons(config_dir):
        configs = []
        for file_name in os.listdir(config_dir):
            file_dir = os.path.join(config_dir, file_name)
            with open(file_dir) as file:
                config = json.load(file)
            configs.append(config)
        return configs
