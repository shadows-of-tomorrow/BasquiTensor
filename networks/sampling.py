import numpy as np


def generate_latent_vectors(latent_dim, n_samples, distribution="gaussian"):
    if distribution == "gaussian":
        return generate_latent_vectors_gaussian(latent_dim, n_samples)
    elif distribution == "ball":
        return generate_latent_vectors_ball(latent_dim, n_samples)
    elif distribution == "bernoulli":
        return generate_latent_vectors_bernoulli(latent_dim, n_samples)
    else:
        ValueError(f"Invalid distribution {distribution}!")


def generate_latent_vectors_gaussian(latent_dim, n_samples):
    z = np.random.normal(size=(n_samples, latent_dim))
    return z.astype('float32')


def generate_latent_vectors_ball(latent_dim, n_samples):
    z = np.random.normal(size=(n_samples, latent_dim))
    r = np.random.uniform(size=(n_samples, 1)) ** (1.0 / latent_dim)
    norm = np.reshape(np.sqrt(np.sum(z ** 2, 1)), newshape=(n_samples, 1))
    return (r * z / norm).astype('float32')


def generate_latent_vectors_bernoulli(latent_dim, n_samples, p_draw=0.50):
    z = np.random.binomial(1, p_draw, size=(n_samples, latent_dim))
    return z.astype('float32')


def generate_real_images(image_processor, n_samples, shape, transform_type="old_to_new"):
    x_real = image_processor.sample_numpy_array(n_samples)
    if transform_type is not None:
        x_real = image_processor.transform_numpy_array(x_real, transform_type)
    if x_real.shape[1] != shape[0]:
        x_real = image_processor.resize_numpy_array(x_real, shape)
    return x_real.astype('float32')


def generate_fake_images(image_processor, generator, n_samples, shape, transform_type=None):
    z = generate_latent_vectors_gaussian(generator.input_shape[1], n_samples)
    x_fake = generator(z, training=False).numpy()
    if transform_type is not None:
        x_fake = image_processor.transform_numpy_array(x_fake, transform_type)
    if x_fake.shape[1] != shape[0]:
        x_fake = image_processor.resize_numpy_array(x_fake, shape)
    return x_fake


def generate_fake_images_from_latents(z_latent, image_processor, generator, shape, transform_type=None):
    x_fake = generator(z_latent, training=False).numpy()
    if transform_type is not None:
        x_fake = image_processor.transform_numpy_array(x_fake, transform_type)
    if x_fake.shape[1] != shape[0]:
        x_fake = image_processor.resize_numpy_array(x_fake, shape)
    return x_fake