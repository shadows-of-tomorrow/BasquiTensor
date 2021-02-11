from gans.composite_constructor import CompositeConstructor
from gans.progressive.generator_constructor import GeneratorConstructorProgressive
from gans.progressive.discriminator_constructor import DiscriminatorConstructorProgressive
from gans.style.generator_constructor import GeneratorConstructorStyle
from gans.style.discriminator_constructor import DiscriminatorConstructorStyle
from gans.style_v2.generator_constructor import GeneratorConstructorStyleV2
from gans.style_v2.discriminator_constructor import DiscriminatorConstructorStyleV2


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
        elif model_type == "StyleGANV2":
            return self._construct_style_gan_v2_networks()
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

    def _construct_style_gan_v2_networks(self):
        discriminators = DiscriminatorConstructorStyleV2(**self.network_config).run()
        generators = GeneratorConstructorStyleV2(**self.network_config).run()
        return discriminators, generators
