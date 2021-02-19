import tensorflow as tf


def generator_loss(discriminator, generator, image_augmenter, z_latent, loss_type):
    # 1. Generate (augmented) fake images with trainable generator.
    x_fake = generator(z_latent, training=True)
    x_fake = image_augmenter.run(x_fake, is_tensor=True)
    # 2. Compute generator loss.
    if loss_type == "vanilla":
        return generator_vanilla_loss(discriminator, x_fake)
    elif loss_type == "wasserstein":
        return generator_wasserstein_loss(discriminator, x_fake)
    else:
        return ValueError(f"Unsupported loss type: {loss_type}")


def discriminator_loss(discriminator, x_real, x_fake, batch_size, loss_type):
    if loss_type == "vanilla":
        return discriminator_vanilla_loss(discriminator, x_real, x_fake)
    elif loss_type == "wasserstein":
        return discriminator_wasserstein_loss(discriminator, x_real, x_fake, batch_size)
    else:
        return ValueError(f"Unsupported loss type: {loss_type}")


def generator_vanilla_loss(discriminator, x_fake):
    # 1. Score fake images.
    y_fake = discriminator(x_fake, training=False)
    # 2. Compute vanilla loss.
    loss_total = tf.reduce_mean(tf.nn.softplus(-y_fake))
    # 3. Gather loss in dictionary.
    loss_dict = {'g_loss_total': loss_total}
    return loss_dict


def generator_wasserstein_loss(discriminator, x_fake):
    # 1. Score fake images.
    y_fake = discriminator(x_fake, training=False)
    # 2. Compute wasserstein loss.
    loss_total = -tf.reduce_mean(y_fake)
    # 3. Gather loss in dictionary.
    loss_dict = {'g_loss_total': loss_total}
    return loss_dict


def discriminator_vanilla_loss(discriminator, x_real, x_fake):
    # 1. Score real and fake images.
    y_real = discriminator(x_real, training=True)
    y_fake = discriminator(x_fake, training=True)
    # 2. Compute vanilla loss.
    loss_real = tf.reduce_mean(tf.nn.softplus(-y_real))
    loss_fake = tf.reduce_mean(tf.nn.softplus(y_fake))
    loss_total = loss_real + loss_fake
    # 3. Compute and add R1 gradient penalty loss.
    gp_loss = r1_gradient_penalty(discriminator, x_real)
    loss_total += gp_loss
    # 4. Gather losses in dictionary.
    loss_dict = {
        'd_loss_total': loss_total,
        'd_loss_real': loss_real,
        'd_loss_fake': loss_fake,
        'd_gp_loss': gp_loss
    }
    return loss_dict, y_real


def discriminator_wasserstein_loss(discriminator, x_real, x_fake, batch_size):
    # 1. Score real and fake images.
    y_real = discriminator(x_real, training=True)
    y_fake = discriminator(x_fake, training=True)
    # 2. Compute wasserstein loss.
    loss_real = -tf.reduce_mean(y_real)
    loss_fake = tf.reduce_mean(y_fake)
    loss_total = loss_real + loss_fake
    # 3. Compute and add gradient penalty loss.
    gp_loss = wasserstein_gradient_penalty(discriminator, x_real, x_fake, batch_size)
    loss_total += gp_loss
    # 4. Compute and add drift penalty loss.
    dp_loss = drift_penalty(y_real)
    loss_total += dp_loss
    # 5. Gather losses in dictionary.
    loss_dict = {
        'd_loss_total': loss_total,
        'd_loss_real': loss_real,
        'd_loss_fake': loss_fake,
        'd_gp_loss': gp_loss,
        'd_dp_loss': dp_loss
    }
    return loss_dict, y_real


def r1_gradient_penalty(discriminator, x_real, r1gp_weight=0.10):
    with tf.GradientTape() as tape:
        tape.watch(x_real)
        y_sum = tf.reduce_sum(discriminator(x_real, training=True))
    gradients = tape.gradient(y_sum, [x_real])[0]
    gradient_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    return (r1gp_weight/2.0) * gradient_penalty


def wasserstein_gradient_penalty(discriminator, x_real, x_fake, batch_size, wgp_weight=10.0):
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    x_int = (1.0 - alpha) * x_real + alpha * x_fake
    with tf.GradientTape() as tape:
        tape.watch(x_int)
        y_int = discriminator(x_int, training=True)
    gradients = tape.gradient(y_int, [x_int])[0]
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)
    return wgp_weight * gradient_penalty


def drift_penalty(y_real, dp_weight=0.001):
    return dp_weight * tf.reduce_mean(tf.square(y_real))
