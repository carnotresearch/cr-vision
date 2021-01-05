"""
Building blocks for segmentation networks
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, utils


def get_dropout_class(dropout_type):
    if dropout_type == "spatial":
        return layers.SpatialDropout2D
    elif dropout_type == "standard":
        return layers.Dropout
    else:
        raise ValueError(
            f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
        )


def unet_conv_block(inputs, 
    filters, # number of filters/output channels
    kernel_size=(3,3), # kernel size
    activation='relu', # activation for conv layers
    padding='same', # padding for conv layers
    kernel_initializer="he_normal",
    use_batch_norm=False,
    dropout=0,
    dropout_type="spatial",
    name='conv'
    ):
    use_bias = not use_batch_norm
    # dropout layer type
    DO = None
    if dropout > 0:
        DO = get_dropout_class(dropout_type)
    # first convolutional block
    net = layers.Conv2D(filters, kernel_size,
        activation=activation, padding=padding,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        name=f'{name}_conv_1')(inputs)
    if use_batch_norm:
        net = layers.BatchNormalization(name=f'{name}_bn')(net)
    if dropout > 0:
        # dropout in-between
        net = DO(dropout, name=f'{name}_dropout')(net)
    # second convolutional block
    net = layers.Conv2D(filters, kernel_size,
        activation=activation, padding=padding,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        name=f'{name}_conv_2')(net)
    return net
