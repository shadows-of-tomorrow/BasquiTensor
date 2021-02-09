import numpy as np

from tensorflow.keras.layers import Input


class DiscriminatorConstructorVanilla:

    def __init__(self, **network_config):
        self.output_res = network_config['output_res']
        self.output_res_log2 = int(np.log2(self.output_res))
        self.n_blocks = self.output_res_log2 - 1

    def run(self):
        input_layer = self._construct_input_layer()

    def _construct_input_layer(self):
        return Input(shape=(self.output_res, self.output_res, 3))


