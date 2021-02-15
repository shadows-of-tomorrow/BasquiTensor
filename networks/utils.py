import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.models import clone_model
from networks.layers import WeightedSum


def update_fade_in(models, step, n_steps):
    alpha = step / float(n_steps - 1)
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)


def update_smoothed_weights(smoothed_model, training_model, alpha=0.999):
    smoothed_weights = smoothed_model.get_weights()
    training_weights = training_model.get_weights()
    for k in range(len(smoothed_weights)):
        smoothed_weights[k] = alpha * smoothed_weights[k] + (1.0 - alpha) * training_weights[k]
    smoothed_model.set_weights(smoothed_weights)


def clone_subclassed_model(model):
    cloned_model = clone_model(model)
    return cloned_model


def generate_real_samples(image_provider, n_samples, shape):
    x = image_provider.sample_batch(n_samples)
    x = image_provider.resize_imgs(x, shape)
    x = np.asarray(x)
    y = np.ones((n_samples, 1))
    return x, y


def generate_latent_points(latent_dim, n_samples):
    z = np.random.normal(size=(n_samples, latent_dim))
    r = np.random.uniform(size=(n_samples, 1)) ** (1.0 / latent_dim)
    norm = np.reshape(np.sqrt(np.sum(z ** 2, 1)), newshape=(n_samples, 1))
    return r * z / norm


def generate_fake_samples(generator, latent_dim, n_samples):
    z = generate_latent_points(latent_dim, n_samples)
    x = generator.predict(z)
    y = -np.ones((n_samples, 1))
    return x, y
