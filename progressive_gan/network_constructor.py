from progressive_gan.discriminator_constructor import DiscriminatorConstructor
from progressive_gan.generator_constructor import GeneratorConstructor
from progressive_gan.composite_constructor import CompositeConstructor


class NetworkConstructor:

    def __init__(self, **network_config):
        self.network_config = network_config

    def run(self):
        discriminators = DiscriminatorConstructor(**self.network_config).run()
        generators = GeneratorConstructor(**self.network_config).run()
        composites = CompositeConstructor(**self.network_config).run(discriminators, generators)
        if self.network_config['use_growing']:
            networks = {
                'discriminators': discriminators,
                'generators': generators,
                'composites': composites
            }
        else:
            networks = {
                'discriminators': [discriminators[-1]],
                'generators': [generators[-1]],
                'composites': [composites[-1]]
            }
        return networks
