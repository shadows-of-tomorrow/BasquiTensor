from tensorflow.keras import backend


def wasserstein_loss(y_true, y_pred):
    """ Computes the vanilla loss. """
    return backend.mean(y_true * y_pred)
