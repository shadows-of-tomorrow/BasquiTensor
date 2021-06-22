from networks.utils.checkpoint_handling import load_network_checkpoint
from networks.stylegan.stylegan_gc import StyleGANGeneratorConstructor
from networks.stylegan.stylegan_dc import StyleGANDiscriminatorConstructor


class NetworkFactory:

    def __init__(self, **network_config):
        self.network_config = network_config

    def run(self):
        networks = self._construct_networks()
        return networks

    def _construct_networks(self):
        if self.network_config['use_checkpoint'] == "True":
            return self._construct_networks_from_checkpoint()
        else:
            return self._construct_networks_from_config()

    def _construct_networks_from_checkpoint(self):
        networks = load_network_checkpoint(self.network_config['checkpoint_location'])
        return networks

    def _construct_networks_from_config(self):
        if self.network_config['model'] == "StyleGAN":
            return self._construct_style_gan_networks()
        else:
            raise ValueError(f"Model type: {self.network_config['model']} is not recognized.")

    def _construct_style_gan_networks(self):
        discriminator = StyleGANDiscriminatorConstructor(**self.network_config).run()
        generator = StyleGANGeneratorConstructor(**self.network_config).run()
        networks = {'discriminator': discriminator, 'generator': generator}
        return networks
