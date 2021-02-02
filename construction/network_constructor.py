from construction.discriminator_constructor import DiscriminatorConstructor
from construction.generator_constructor import GeneratorConstructor
from construction.composite_constructor import CompositeConstructor


class NetworkConstructor:

    def __init__(self, **network_config):
        self.latent_dim = network_config['latent_dim']
        self.max_filters = network_config['max_filters']
        self.output_res = network_config['output_res']

    def run(self):
        discriminators = DiscriminatorConstructor(self.output_res, self.max_filters).run()
        generators = GeneratorConstructor(self.output_res, self.max_filters, self.latent_dim).run()
        composites = CompositeConstructor().run(discriminators, generators)
        networks = {'discriminators': discriminators, 'generators': generators, 'composites': composites}
        return networks
