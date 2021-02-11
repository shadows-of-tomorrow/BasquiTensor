from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from gans.loss_functions import wasserstein_loss


class CompositeConstructor:

    def __init__(self, **network_config):
        self.adam_params = network_config['adam_params']

    def run(self, discriminators, generators):
        assert len(discriminators) == len(generators)
        # 1. Initialize list of composites.
        composites = []
        # 2. Create composites.
        for k in range(len(discriminators)):
            # 2.1 Extract discriminator and generator models at current stage.
            ds, gs = discriminators[k], generators[k]
            # 2.2 Construct next "tuning" composite.
            ds[0].trainable = False
            tuning_composite = Sequential()
            tuning_composite.add(gs[0])
            tuning_composite.add(ds[0])
            self._compile_model(tuning_composite)
            # 2.3 Construct next "fade-in" composite.
            ds[1].trainable = False
            fade_in_composite = Sequential()
            fade_in_composite.add(gs[1])
            fade_in_composite.add(ds[1])
            self._compile_model(fade_in_composite)
            # 2.4 Append GAN models to list.
            composites.append([tuning_composite, fade_in_composite])
        return composites

    def _compile_model(self, model):
        lr = self.adam_params['lr']
        beta_1 = self.adam_params['beta_1']
        beta_2 = self.adam_params['beta_2']
        epsilon = self.adam_params['epsilon']
        model.compile(loss=wasserstein_loss, optimizer=Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon))
