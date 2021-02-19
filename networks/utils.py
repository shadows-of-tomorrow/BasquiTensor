import numpy as np
import tensorflow as tf
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


def generate_latent_vectors(latent_dim, n_samples, dtype='float32'):
    z = np.random.normal(size=(n_samples, latent_dim))
    r = np.random.uniform(size=(n_samples, 1)) ** (1.0 / latent_dim)
    norm = np.reshape(np.sqrt(np.sum(z ** 2, 1)), newshape=(n_samples, 1))
    return (r * z / norm).astype(dtype)


def generate_fake_samples(image_processor, generator, n_samples, shape, dtype='float32', transform_type=None):
    z = generate_latent_vectors(generator.input_shape[1], n_samples, dtype)
    x_fake = generator(z, training=False).numpy()
    if transform_type is not None:
        x_fake = image_processor.transform_numpy_array(x_fake, transform_type)
    if x_fake.shape[1] != shape[0]:
        x_fake = image_processor.resize_numpy_array(x_fake, shape)
    return x_fake


def generate_real_samples(image_processor, n_samples, shape, dtype='float32', transform_type="old_to_new"):
    x_real = image_processor.sample_numpy_array(n_samples)
    if transform_type is not None:
        x_real = image_processor.transform_numpy_array(x_real, transform_type)
    if x_real.shape[1] != shape[0]:
        x_real = image_processor.resize_numpy_array(x_real, shape)
    return x_real.astype(dtype)


def tensor_dict_to_numpy(tensor_dict):
    for key, value in tensor_dict.items():
        if isinstance(value, tf.Tensor):
            tensor_dict[key] = value.numpy()
    return tensor_dict
