import os
from processing.image_provider import ImageProvider
from training.trainer import GANTrainer
from networks.discriminators import DiscriminatorCreator
from networks.generators import GeneratorCreator
from networks.composites import CompositeCreator

if __name__ == "__main__":

    # 1. Construct networks.
    discriminators = DiscriminatorCreator().execute()
    generators = GeneratorCreator().execute()
    composites = CompositeCreator().execute(discriminators, generators)

    # 2. Construct image handler.
    dir_in = os.path.join(os.path.dirname(__file__), 'images', 'in')
    image_provider = ImageProvider(dir_in)

    # 3. Define training parameters.
    latent_dim = 256
    n_batches = [16, 16, 16, 8, 8, 4, 4]
    n_epochs = [10, 10, 10, 10, 10, 10, 10]

    # 4. Train networks.
    trainer = GANTrainer(latent_dim, image_provider, n_batches, n_epochs)
    trainer.execute(discriminators, generators, composites)
