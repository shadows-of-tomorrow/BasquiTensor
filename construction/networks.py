from construction.discriminators import DiscriminatorConstructor
from construction.generators import GeneratorConstructor
from construction.composites import CompositeConstructor


class NetworkConstructor:

    def __init__(self, **network_config):
        self.latent_dim = network_config['latent_dim']
        self.max_filters = network_config['max_filters']
        self.output_res = network_config['output_res']

    def run(self):
        discriminators = DiscriminatorConstructor(max_filters=self.max_filters,
                                                  output_res=self.output_res).run()

        generators = GeneratorConstructor(latent_dim=self.latent_dim,
                                          max_filters=self.max_filters,
                                          output_res=self.output_res).run()

        composites = CompositeConstructor().run(discriminators=discriminators,
                                                generators=generators)

        networks = {'discriminators': discriminators,
                    'generators': generators,
                    'composites': composites}
        return networks
