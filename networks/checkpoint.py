import os
from tensorflow.keras.models import load_model
from networks.layers import DenseEQL
from networks.layers import Conv2DEQL
from networks.layers import MinibatchStDev
from networks.layers import Constant
from networks.layers import NoiseModulation
from networks.layers import AdaptiveInstanceModulation
from networks.stylegan.stylegan_generator import StyleGANGenerator
from networks.stylegan.stylegan_discriminator import StyleGANDiscriminator

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


def load_network_checkpoint(network_path):
    # 1. Read fields of custom subclassed Keras models.
    fields = read_custom_fields(network_path)
    # 2. Read pickled (.h5) networks.
    discriminator = load_model(os.path.join(network_path, 'discriminator.h5'), CUSTOM_OBJECTS)
    generator = load_model(os.path.join(network_path, 'generator.h5'), CUSTOM_OBJECTS)
    generator_smoothed = load_model(os.path.join(network_path, 'generator_smoothed.h5'), CUSTOM_OBJECTS)
    # 3. Convert pickled networks to corresponding custom model class.
    discriminator = convert_and_compile_network(discriminator, fields, "StyleGANDiscriminator", True)
    generator = convert_and_compile_network(generator, fields, "StyleGANGenerator", True)
    generator_smoothed = convert_and_compile_network(generator_smoothed, fields, "StyleGANGenerator", False)
    # 4. Convert networks to correct format.
    networks = convert_networks_to_checkpoint_format(discriminator, generator, generator_smoothed)
    return networks


def convert_networks_to_checkpoint_format(discriminator, generator, generator_smoothed):
    networks = {
        'discriminators': [[discriminator, None]],
        'generators': [[generator, None]],
        'generators_smoothed': [[generator_smoothed, None]]
    }
    return networks


def convert_and_compile_network(network, fields, network_type, compile):
    if network_type == "StyleGANDiscriminator":
        discriminator = StyleGANDiscriminator(network.input, network.output)
        if compile:
            discriminator.compile(optimizer=network.optimizer)
        discriminator.loss_type = fields['loss_type']
        discriminator.ada_target = fields['ada_target']
        discriminator.ada_smoothing = fields['ada_smoothing']
        return discriminator
    elif network_type == "StyleGANGenerator":
        generator = StyleGANGenerator(network.input, network.output)
        if compile:
            generator.compile(optimizer=network.optimizer)
        generator.loss_type = fields['loss_type']
        generator.latent_dist = fields['latent_dist']
        return generator
    else:
        ValueError(f"Network type {network_type} not recognized.")


def read_custom_fields(network_path):
    return read_txt_file(network_path, 'fields.txt')[-1]


# -------------------- Move this to util class at some point. ----------------------------------------------------------
def read_txt_file(loss_dir, name):
    with open(loss_dir + '/' + name, 'r') as file:
        lines = file.readlines()
        losses = []
    for line in lines:
        line = parse_line(line)
        losses.append(line)
    return losses


def parse_line(line):
    line = line.split(",")[:-1]
    line = dict([item.split(':') for item in line])
    for key, value in line.items():
        value = str_to_float(value)
        line[key] = value
    return line


def str_to_float(string):
    try:
        return float(string)
    except ValueError:
        return string

# ----------------------------------------------------------------------------------------------------------------------
