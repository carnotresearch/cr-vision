import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend, models, utils

from .utils import deduce_input_shape




def conv2d(input,
    filters,
    rows,
    columns,
    padding='same',
    strides=(1,1),
    name='conv',
    data_format='channels_last',
    batch_normalization=True,
    activation='relu'):
    """
    Constructions a convolutional layer with batch normalization
    """
    net = layers.Conv2D(
        filters,
        (rows, columns),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=name + '_conv',
        data_format=data_format)(input)
    ch_axis = get_channels_axis(data_format)
    # Add batch normalization
    if batch_normalization:
        net = layers.BatchNormalization(axis=ch_axis, 
            scale=False, name=name + '_bn')(net)
    # Add activation
    if activation:
        net = layers.Activation(activation, name=name)(net)
    # Return the combined network
    return net



def stem_v3(input, data_format="channels_last"):
    # input 299x299x3 output: 149x149x32
    net = conv2d(input, filters=32, rows=3, columns=3, 
        strides=(2, 2), padding='valid')
