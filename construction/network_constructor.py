from construction.discriminator_constructor import DiscriminatorConstructor
from construction.generator_constructor import GeneratorConstructor
from construction.composite_constructor import CompositeConstructor


class NetworkConstructor:

    def __init__(self, **network_config):
        self.network_config = network_config

    def run(self):
        discriminators = DiscriminatorConstructor(**self.network_config).run()
        generators = GeneratorConstructor(**self.network_config).run()
        composites = CompositeConstructor().run(discriminators, generators)
        networks = {'discriminators': discriminators, 'generators': generators, 'composites': composites}
        return networks
