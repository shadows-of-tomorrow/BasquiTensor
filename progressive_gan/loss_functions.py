from tensorflow.keras import backend


def wasserstein_loss(y_true, y_pred):
    """ Computes the wasserstein loss. """
    return backend.mean(y_true * y_pred)
