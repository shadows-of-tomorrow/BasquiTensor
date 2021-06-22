from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from tensorflow.keras.models import clone_model
from networks.utils.custom_layers import Constant, WeightedSum
from networks.utils.custom_layers import DenseEQL
from networks.utils.custom_layers import Conv2DEQL
from networks.utils.custom_layers import NoiseModulation
from networks.utils.custom_layers import MinibatchStDev
from networks.utils.custom_layers import AdaptiveInstanceModulation
from networks.stylegan.stylegan_g import StyleGANGenerator
from networks.stylegan.stylegan_d import StyleGANDiscriminator

CUSTOM_OBJECTS = {
    'StyleGANDiscriminator': StyleGANDiscriminator,
    'StyleGANGenerator': StyleGANGenerator,
    'DenseEQL': DenseEQL,
    'Conv2DEQL': Conv2DEQL,
    "MinibatchStDev": MinibatchStDev,
    "Constant": Constant,
    "NoiseModulation": NoiseModulation,
    "AdaptiveInstanceModulation": AdaptiveInstanceModulation
}


def load_model_from_disk(dir_model):
    return load_model(dir_model, CUSTOM_OBJECTS)


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
