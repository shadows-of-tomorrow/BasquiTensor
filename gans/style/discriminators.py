from gans.progressive.discriminators import DiscriminatorConstructorProgressive


class DiscriminatorConstructorStyle(DiscriminatorConstructorProgressive):

    def __init__(self, **network_config):
        super().__init__(**network_config)
