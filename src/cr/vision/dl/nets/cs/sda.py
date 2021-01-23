"""
Stacked denoising auto-encoders
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, backend, models, utils



def cnn_sda(inputs, kernel_initializer="he_normal"):
    num_color_channels = inputs.shape[-1]
    kernel_size = 16
    strides = kernel_size
    patch_size = kernel_size * kernel_size * num_color_channels
    measurements = patch_size / 4
    # note that the sensing layer is not trainable
    # from N samples in an image patch to M measurements (as channels)
    net = layers.Conv2D(
        filters=measurements, 
        kernel_size=kernel_size, 
        strides=strides,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        trainable=False,
        name="sensor")(inputs)
    # each output channel is one measurement
    # the filter applies to whole patch in one go
    # we will now use 1x1 convolutions for following layers
    # From M measurements (arranged in channels) to N samples (arranged in channels)
    net = layers.Conv2D(patch_size, 1, 
        use_bias=True,
        kernel_initializer=kernel_initializer,
        activation="relu",
        name="layer-1")(net)
    net = layers.Conv2D(measurements, 1, 
        use_bias=True,
        kernel_initializer=kernel_initializer,
        activation="relu",
        name="layer-2")(net)
    # This layer is the inverse of the first layer.
    # It converts a strip over samples on the
    # channels dimension into an image patch
    net = layers.Conv2DTranspose(
        filters=num_color_channels, 
        kernel_size=kernel_size,
        strides=strides, 
        use_bias=True,
        kernel_initializer=kernel_initializer,
        activation="sigmoid",
        name="layer-3")(net)
    return net


def model_sda(input_shape):
    inputs = layers.Input(input_shape, name='input')
    net = cnn_sda(inputs)
    model = models.Model(inputs=[inputs], outputs=[net])
    return model
    