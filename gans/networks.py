from gans.composites import CompositeConstructor
from gans.progressive.generators import GeneratorConstructor
from gans.progressive.discriminators import DiscriminatorConstructor


class NetworkFactory:

    def __init__(self, **network_config):
        self.network_config = network_config

    def run(self):
        discriminators = DiscriminatorConstructor(**self.network_config).run()
        generators = GeneratorConstructor(**self.network_config).run()
        composites = CompositeConstructor(**self.network_config).run(discriminators, generators)
        return {'discriminators': discriminators, 'generators': generators, 'composites': composites}
