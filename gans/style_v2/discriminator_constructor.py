from gans.progressive.discriminator_constructor import DiscriminatorConstructorProgressive


class DiscriminatorConstructorStyleV2(DiscriminatorConstructorProgressive):

    def __init__(self, skip_layers=3, **network_config):
        super().__init__(skip_layers, **network_config)

    def run(self):
        discriminators = super().run()
        return [[discriminators[-1][0]] * 2]
