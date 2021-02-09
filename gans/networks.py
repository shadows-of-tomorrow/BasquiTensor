from gans.composites import CompositeConstructor

from gans.progressive.generators import GeneratorConstructorProgressive
from gans.progressive.discriminators import DiscriminatorConstructorProgressive

from gans.style.generators import GeneratorConstructorStyle
from gans.style.discriminators import DiscriminatorConstructorStyle


class NetworkConstructor:

    def __init__(self, **network_config):
        self.network_config = network_config

    def run(self):
        discriminators, generators = self._construct_gan_networks()
        composites = CompositeConstructor(**self.network_config).run(discriminators, generators)
        return {'discriminators': discriminators, 'generators': generators, 'composites': composites}

    def _construct_gan_networks(self):
        model_type = self.network_config['model']
        if model_type == "ProGAN":
            return self._construct_progressive_gan_networks()
        elif model_type == "StyleGAN":
            return self._construct_style_gan_networks()
        else:
            raise Exception(f"Model type: {model_type} is not recognized!")

    def _construct_progressive_gan_networks(self):
        discriminators = DiscriminatorConstructorProgressive(**self.network_config).run()
        generators = GeneratorConstructorProgressive(**self.network_config).run()
        return discriminators, generators

    def _construct_style_gan_networks(self):
        discriminators = DiscriminatorConstructorStyle(**self.network_config).run()
        generators = GeneratorConstructorStyle(**self.network_config).run()
        return discriminators, generators
