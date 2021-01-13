"""
U-Net models
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, backend, models, utils

# helper functions
from .utils import get_center_crop_location
# block of 2 convolution layers
from .blocks import unet_conv_block

def cnn_unet(inputs, 
    filters=64,
    num_layers=4):

    # array to hold outputs of blocks
    # in the contracting path
    # for later use
    # in the expanding path
    down_blocks = []

    # build the contracting path
    # Create the downsampling layers
    net = inputs
    for i in range(num_layers):
        name = f'contract_{i+1}'
        net = unet_conv_block(inputs=net,
            filters=filters,
            padding='valid',
            name=name)
        # save this layer for later reference
        down_blocks.append(net)
        # add pooling
        net = layers.MaxPooling2D((2,2), strides=2,
            name=f'{name}_pool')(net)
        # increase the number of filters
        filters = filters * 2

    # build the lateral path
    net = unet_conv_block(inputs=net, 
        filters=filters,
        padding='valid',
        name='lateral')

    # build the expanding path
    i = num_layers
    for conv in reversed(down_blocks):
        # decrease the number of filters
        filters = filters // 2
        name = f'expand_{i}'
        # upsample 
        net = layers.Conv2DTranspose(filters, (2,2),
            strides=(2,2),
            padding='valid',
            name=f'{name}_upconv')(net)
        # identify the center crop area
        center_crop = get_center_crop_location(
            source=conv, 
            destination=net)
        # perform cropping
        conv = layers.Cropping2D(cropping=center_crop,
            name=f'{name}_crop')(conv)
        # concatenate 
        net = layers.concatenate([net, conv], name=f'{name}_concatenate')
        # add one more convolutional block
        net = unet_conv_block(inputs=net, 
            filters=filters,
            padding='valid',
            name=name)
        i = i - 1
    # we are done
    return net


def model_unet_ronneberger_2015(input_shape,
    classes=2,
    output_activation='softmax'):
    inputs = layers.Input(input_shape, name='input')
    net = cnn_unet(inputs)
    # build the output layer using 1x1 convolution
    outputs = layers.Conv2D(classes, (1,1), 
        activation=output_activation, name='output')(net)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model



def cnn_custom_unet(inputs, 
    filters=16,
    num_layers=4,
    activation='relu',
    use_batch_norm=True,
    use_bias=None,
    upsample_mode='deconv',
    dropout=0.3,
    dropout_change_per_layer=0,
    dropout_type='spatial',
    use_dropout_on_upsampling=False):
    # array to hold outputs of blocks
    # in the contracting path
    # for later use
    # in the expanding path
    down_blocks = []

    # build the contracting path
    # Create the downsampling layers
    net = inputs
    for i in range(num_layers):
        name = f'contract_{i+1}'
        net = unet_conv_block(inputs=net,
            filters=filters,
            padding='same',
            activation=activation,
            use_batch_norm=use_batch_norm,
            use_bias=use_bias,
            dropout=dropout,
            dropout_type=dropout_type,
            name=name)
        # save this layer for later reference
        down_blocks.append(net)
        # add pooling
        net = layers.MaxPooling2D((2,2), strides=2,
            name=f'{name}_pool')(net)
        # increase the number of filters
        filters = filters * 2
        # increase dropout if required
        dropout = dropout + dropout_change_per_layer

    # build the lateral path
    net = unet_conv_block(inputs=net, 
        filters=filters,
        padding='same',
        activation=activation,
        use_batch_norm=use_batch_norm,
        use_bias=use_bias,
        dropout=dropout,
        dropout_type=dropout_type,
        name='lateral')

    if not use_dropout_on_upsampling:
        # disable dropout for expansion part of network
        dropout = 0
        dropout_change_per_layer = 0

    # build the expanding path
    i = num_layers
    for conv in reversed(down_blocks):
        # decrease the number of filters
        filters = filters // 2
        name = f'expand_{i}'
        # decrease dropout if required
        dropout -= dropout_change_per_layer
        # upsample
        if upsample_mode == 'deconv': 
            net = layers.Conv2DTranspose(filters, 
                (2,2),
                strides=(2,2),
                padding='same',
                name=f'{name}_upconv')(net)
        else:
            net = layers.UpSampling2D(
                strides=(2,2),
                name=f'{name}_upconv')(net)
        # cropping is not needed due to same padding
        # TODO attention concatenation
        # concatenate 
        net = layers.concatenate([net, conv], name=f'{name}_concatenate')
        # add one more convolutional block
        net = unet_conv_block(inputs=net, 
            filters=filters,
            padding='same',
            activation=activation,
            use_batch_norm=use_batch_norm,
            use_bias=use_bias,
            dropout=dropout,
            dropout_type=dropout_type,
            name=name)
        i = i - 1
    # we are done
    return net


def model_custom_unet(input_shape,
    classes=1,
    output_activation='sigmoid',
    filters=16,
    num_layers=4,
    activation='relu',
    use_batch_norm=True,
    use_bias=None,
    upsample_mode='deconv',
    dropout=0.3,
    dropout_change_per_layer=0,
    dropout_type='spatial',
    use_dropout_on_upsampling=False):
    inputs = layers.Input(input_shape, name='input')
    net = cnn_custom_unet(inputs,
        filters=filters,
        num_layers=num_layers,
        activation=activation,
        use_batch_norm=use_batch_norm,
        use_bias=use_bias,
        upsample_mode=upsample_mode,
        dropout=dropout,
        dropout_change_per_layer=dropout_change_per_layer,
        dropout_type=dropout_type,
        use_dropout_on_upsampling=use_dropout_on_upsampling)
    # build the output layer using 1x1 convolution
    outputs = layers.Conv2D(classes, (1,1), 
        activation=output_activation, name='output')(net)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model
