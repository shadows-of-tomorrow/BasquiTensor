from progressive_gan.loss_functions import wasserstein_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class CompositeConstructor:
    """ Creates a list of progressively growing composites. """
    def __init__(self, **network_config):
        self.adam_params = network_config['adam_params']

    def run(self, discriminators, generators):
        assert len(discriminators) == len(generators)
        # 1. Initialize list of GANs.
        composites = []
        # 2. Create GANs.
        for k in range(len(discriminators)):
            # 2.1 Extract discriminator and generator models at current stage.
            ds, gs = discriminators[k], generators[k]
            # 2.2 Construct next GAN model.
            ds[0].trainable = False
            next_model = Sequential()
            next_model.add(gs[0])
            next_model.add(ds[0])
            self._compile_model(next_model)
            # 2.3 Construct fade-in GAN model.
            ds[1].trainable = False
            fade_in_model = Sequential()
            fade_in_model.add(gs[1])
            fade_in_model.add(ds[1])
            self._compile_model(fade_in_model)
            # 2.4 Append GAN models to list.
            composites.append([next_model, fade_in_model])
        return composites

    def _compile_model(self, model):
        lr = self.adam_params['lr']
        beta_1 = self.adam_params['beta_1']
        beta_2 = self.adam_params['beta_2']
        epsilon = self.adam_params['epsilon']
        model.compile(loss=wasserstein_loss, optimizer=Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon))
