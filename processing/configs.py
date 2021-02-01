import os
import json


class ConfigProcessor:
    """ Handles the reading and processing of configuration files. """
    def __init__(self, from_json=True):
        self.from_json = from_json

    def read_configs(self, config_dir):
        if self.from_json is True:
            return self._read_jsons(config_dir)

    @staticmethod
    def _read_jsons(config_dir):
        configs = []
        for file_name in os.listdir(config_dir):
            file_dir = os.path.join(config_dir, file_name)
            with open(file_dir) as file:
                config = json.load(file)
            configs.append(config)
        return configs
