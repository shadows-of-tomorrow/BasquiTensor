from gans.composites import CompositeConstructor

from gans.progressive.generators import ProGANGeneratorConstructor
from gans.progressive.discriminators import ProGANDiscriminatorConstructor

from gans.style.generators import StyleGANGeneratorConstructor
from gans.style.discriminators import StyleGANDiscriminatorConstructor


class NetworkFactory:

    def __init__(self, **network_config):
        self.network_config = network_config

    def run(self):
        model_type = self.network_config['model']
        if model_type == "ProGAN":
            discriminators, generators = self._construct_progressive_gan_networks()
        elif model_type == "StyleGAN":
            discriminators, generators = self._construct_style_gan_networks()
        else:
            raise Exception(f"Model type: {model_type} is not recognized!")
        composites = CompositeConstructor(**self.network_config).run(discriminators, generators)
        return {'discriminators': discriminators, 'generators': generators, 'composites': composites}

    def _construct_progressive_gan_networks(self):
        discriminators = ProGANDiscriminatorConstructor(**self.network_config).run()
        generators = ProGANGeneratorConstructor(**self.network_config).run()
        return discriminators, generators

    def _construct_style_gan_networks(self):
        discriminators = StyleGANDiscriminatorConstructor(**self.network_config).run()
        generators = StyleGANGeneratorConstructor(**self.network_config).run()
        return discriminators, generators
