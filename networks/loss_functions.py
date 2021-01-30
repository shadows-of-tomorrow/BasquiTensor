from tensorflow.keras import backend


def wasserstein_loss_drift_penalty(y_true, y_pred, epsilon=0.001):
    """ Computes the wasserstein loss with a drift penalty. """
    return wasserstein_loss(y_true, y_pred) + epsilon * backend.mean(backend.square(y_pred))


def wasserstein_loss(y_true, y_pred):
    """ Computes the wasserstein loss. """
    return backend.mean(y_true * y_pred)
