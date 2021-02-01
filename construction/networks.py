from construction.discriminators import DiscriminatorConstructor
from construction.generators import GeneratorConstructor
from construction.composites import CompositeConstructor


class NetworkConstructor:

    def __init__(self, **network_config):
        self.latent_dim = network_config['latent_dim']

    def execute(self):
        discriminators = DiscriminatorConstructor().execute()
        generators = GeneratorConstructor(latent_dim=self.latent_dim).execute()
        composites = CompositeConstructor().execute(discriminators, generators)
        networks = {'discriminators': discriminators,
                    'generators': generators,
                    'composites': composites}
        return networks
