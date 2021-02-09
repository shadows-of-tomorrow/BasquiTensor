from gans.progressive.discriminators import ProGANDiscriminatorConstructor


class StyleGANDiscriminatorConstructor(ProGANDiscriminatorConstructor):

    def __init__(self, **network_config):
        super().__init__(**network_config)
