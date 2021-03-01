from networks.stylegan.stylegan_generator_constructor import StyleGANGeneratorConstructor
from networks.stylegan.stylegan_discriminator_constructor import StyleGANDiscriminatorConstructor


class NetworkFactory:

    def __init__(self, **network_config):
        self.network_config = network_config

    def run(self):
        discriminators, generators = self._construct_networks()
        return {'discriminators': discriminators, 'generators': generators}

    def _construct_networks(self):
        model_type = self.network_config['model']
        if model_type == "StyleGAN":
            return self._construct_style_gan_networks()
        else:
            raise Exception(f"Model type: {model_type} is not recognized!")

    def _construct_style_gan_networks(self):
        discriminators = StyleGANDiscriminatorConstructor(**self.network_config).run()
        generators = StyleGANGeneratorConstructor(**self.network_config).run()
        return discriminators, generators
