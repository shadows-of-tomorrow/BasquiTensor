from gans.progressive.discriminator_constructor import DiscriminatorConstructorProgressive


class DiscriminatorConstructorStyle(DiscriminatorConstructorProgressive):

    def __init__(self, **network_config):
        super().__init__(**network_config)
