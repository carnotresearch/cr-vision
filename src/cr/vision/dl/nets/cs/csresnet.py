"""
Conv CS Net
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, backend, models, utils

from dataclasses import dataclass

@dataclass
class CSResNet:
    encoder: models.Model = None
    decoder: models.Model = None
    autoencoder: models.Model = None


def bulid_encoder(image, 
    patch_size=16,
    stride_size=None,
    compression_ratio=64,
    kernel_initializer="he_normal"
    ):
    num_color_channels = image.shape[-1]
    patch_volume = patch_size * patch_size * num_color_channels
    measurements = patch_volume // compression_ratio
    if stride_size is None:
        stride_size = patch_size
    #print(f'cr: {compression_ratio}, c: {num_color_channels}, pv: {patch_volume} k : {patch_size}, s: {stride_size}: m: {measurements}')
    net = layers.Conv2D(
        filters=measurements, 
        kernel_size=patch_size, 
        strides=stride_size,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        trainable=True,
        name="sensor")(image)
    return net


def residual_block(inputs, n_block):
    in_channels = inputs.shape[-1]
    out_channels = 96
    residual = layers.Conv2D(
        filters=out_channels, 
        kernel_size=3, 
        use_bias=True,
        activation="relu",
        padding="same",
        name=f"residual-{n_block}")(inputs)

    if in_channels == out_channels:
        result = layers.Add(name=f'add-{n_block}')([inputs, residual])
    else:
        result = residual
    return result


def build_decoder(encoded_inputs,
    patch_size=16,
    stride_size=None,
    num_color_channels=3,
    compression_ratio=4,
    kernel_initializer="he_normal"):
    patch_volume = patch_size * patch_size * num_color_channels
    measurements = patch_volume // compression_ratio
    if stride_size is None:
        stride_size = patch_size
    net = encoded_inputs
    # # Following network achieves the image reconstruction
         
    # # each output channel is one measurement
    # # the filter applies to whole patch in one go
    # # we will now use 1x1 convolutions for following layers
    # # From M measurements (arranged in channels) to N samples (arranged in channels)
    # net = layers.Conv2D(
    #     filters=patch_size, 
    #     kernel_size=1, 
    #     use_bias=True,
    #     kernel_initializer=kernel_initializer,
    #     activation="relu",
    #     name="layer-1")(encoded_inputs)
    # net = layers.BatchNormalization(name=f'bn_1')(net)


    # net = layers.Conv2D(
    #     filters=measurements, 
    #     kernel_size=1, 
    #     use_bias=True,
    #     kernel_initializer=kernel_initializer,
    #     activation="relu",
    #     name="layer-2")(net)
    # net = layers.BatchNormalization(name=f'bn_2')(net)

    # This layer is the inverse of the first layer.
    # It converts a strip over samples on the
    # channels dimension into an image patch
    net = layers.Conv2DTranspose(
        filters=num_color_channels, 
        kernel_size=patch_size,
        strides=stride_size, 
        use_bias=True,
        kernel_initializer=kernel_initializer,
        activation="relu",
        name="initial")(net)

    # add a few residual blocks
    for n in range(4):
        net = residual_block(net, n+1)

    # add a final block to reconstruct the image
    net = layers.Conv2D(
        filters=num_color_channels, 
        kernel_size=3, 
        use_bias=True,
        activation="relu",
        padding="same",
        name=f"final")(net)
    return net



def build_models(input_shape,
    patch_size=16,
    compression_ratio=64,
    kernel_initializer="he_normal"
    ):
    num_color_channels = input_shape[-1]

    # Encoder Model
    inputs = layers.Input(input_shape, name='input')
    encoder_net = bulid_encoder(inputs,
        patch_size=patch_size,
        compression_ratio=compression_ratio,
        kernel_initializer=kernel_initializer
        )
    encoder = models.Model(inputs, encoder_net, name='CSResNet_Encoder')

    # Decoder Model
    # This is our encoded (32-dimensional) input
    encoded_shape = encoder_net.shape[1:]
    encoded_input = keras.Input(shape=encoded_shape, name="encoded_input")
    decoder_net = build_decoder(encoded_input,
        patch_size=patch_size,
        num_color_channels=num_color_channels,
        compression_ratio=compression_ratio,
        kernel_initializer=kernel_initializer
        )
    decoder = models.Model(encoded_input, decoder_net, name='CSResNet_Decoder')

    # Autoencoder Model
    encoder_layer = encoder(inputs)
    encoder_layer.trainable = False
    decoder_layer = decoder(encoder_layer)
    autoencoder = models.Model(inputs, decoder_layer, 
        name='CSResNet_Autoencoder')
    
    model = CSResNet(encoder=encoder, decoder=decoder, autoencoder=autoencoder)
    return model