import numpy as np
from proGAN.layers import WeightedSum
from tensorflow.keras import backend


def update_fade_in(models, step, n_steps):
    """ Updates the alpha parameter in the WeightedSum layers. """
    # 1. Compute new alpha.
    alpha = step / float(n_steps - 1)
    # 2. Update alpha for all models.
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)


def generate_real_samples(image_provider, n_samples, shape):
    """ Generates an x, y pair of real examples. """
    x = image_provider.sample_batch(n_samples)
    x = image_provider.scale_imgs(x, shape)
    x = np.asarray(x)
    y = np.ones((n_samples, 1))
    return x, y


def generate_latent_points(latent_dim, n_samples):
    """ Draws random samples from an n-dimensional ball. """
    z = np.random.normal(size=(n_samples, latent_dim))
    r = np.random.uniform(size=(n_samples, 1)) ** (1.0 / latent_dim)
    norm = np.reshape(np.sqrt(np.sum(z**2, 1)), newshape=(n_samples, 1))
    return r * z / norm


def generate_fake_samples(generator, latent_dim, n_samples):
    """ Generates an (x,y) pair of fake examples. """
    z = generate_latent_points(latent_dim, n_samples)
    x = generator.predict(z)
    y = -np.ones((n_samples, 1))
    return x, y
