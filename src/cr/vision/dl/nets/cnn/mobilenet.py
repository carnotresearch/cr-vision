"""
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, backend, models, utils

from .utils import deduce_input_shape, get_channels_axis
from cr.vision.dl.nets.publicweights import FCHOLLET



def input_block(img_input, filters=32, alpha=1, 
    kernel_size=3, stride=2, data_format="channels_last"):
    """
    First convolution layer with batch normalization and relu 6 to
    process input image
    """
    # channel axis for batch normalization
    ch_axis = get_channels_axis(data_format)
    # number of output channels depends on alpha
    filters = int(filters * alpha)
    # since we have a stride of 2, hence 
    # an asymmetric padding of 1 pixels on two sides is enough 
    # input is (224 x 224 x 3)
    # output is (225 x 225 x 3)
    net = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), 
        name='conv1_pad')(img_input)
    # first 3x3 convolution layer with stride of 2
    # We don't use bias
    # since padding has already been added, we go with valid padding
    # input is (225 x 225 x 3)
    # output is (112 x 112 x 32)
    net = layers.Conv2D(filters, kernel_size,
                      padding='valid',
                      use_bias=False,
                      strides=stride,
                      name='conv1')(net)
    # add batch normalization to the output of conv layer
    net = layers.BatchNormalization(axis=ch_axis, 
        name='conv1_bn')(net)
    # add activation relu with a max value of 6
    net = layers.ReLU(6., name='conv1_relu')(net)
    return net



def depthwise_conv_block(input, filters, alpha=1, 
    depth_multiplier=1, stride=1, block_id=1, data_format="channels_last"):
    # channel axis for batch normalization
    ch_axis = get_channels_axis(data_format)
    # number of output channels depends on alpha
    filters = int(filters * alpha)
    if stride == 1:
        # we will use same padding strategy in the depth-wise conv
        padding = 'same'
        net = input
    else:
        # let's add a padding of 1 pixels on two sides.
        net = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)(input)
        # now we can use valid padding for down-conversion by stride 2
        padding = 'valid'
    # time to apply depth wise convolution layer
    net = layers.DepthwiseConv2D((3, 3),
                               padding=padding,
                               depth_multiplier=depth_multiplier,
                               strides=stride,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(net)
    # add batch normalization
    net = layers.BatchNormalization(
        axis=ch_axis, name='conv_dw_%d_bn' % block_id)(net)
    # add activation relu with a max value of 6
    net = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(net)
    # we will now apply point-wise convolution across input channels
    net = layers.Conv2D(filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(net)
    # add batch normalization
    net = layers.BatchNormalization(axis=ch_axis,
                                  name='conv_pw_%d_bn' % block_id)(net)
    # add activation relu with a max value of 6
    net = layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(net)
    return net


def top_block(net, classes=1000, alpha=1, dropout=1e-3, data_format="channels_last"):
    """
    Top block to process the CNN output for classification
    """
    # shape of channels as a 1x1 image
    input_channels = int(1024 * alpha)
    if data_format == 'channels_first':
        shape = (input_channels, 1, 1)
    else:
        shape = (1, 1, input_channels)
    # Apply global average pooling (7x7 to 1x1)
    net = layers.GlobalAveragePooling2D()(net)
    # Reshape for dropout
    net = layers.Reshape(shape, name='reshape_1')(net)
    # apply dropout
    net = layers.Dropout(dropout, name='dropout')(net)
    # The FC input 1024 channels => output 1000 classes
    # The FC layer can be thought of as 1x1 convolution
    # Change from 1024 channels to 1000 channels
    net = layers.Conv2D(classes, (1, 1),
                      padding='same',
                      name='conv_preds')(net)
    # reshape again from 1x1 image to feature vector of size 1000
    net = layers.Reshape((classes,), name='reshape_2')(net)
    # Apply softmax activation to get probability vector
    net = layers.Activation('softmax', name='act_softmax')(net)
    return net

def pool_block(net, pooling, data_format):
    if pooling == 'avg':
        net = layers.GlobalAveragePooling2D(name='avg_pool',
            data_format=data_format)(net)
    elif pooling == 'max':
        net = layers.GlobalMaxPooling2D(name='max_pool',
            data_format=data_format)(net)
    return net


def cnn_mobilenet(img_input,
    alpha=1.0, 
    depth_multiplier=1,
    data_format="channels_last"):
    net = input_block(img_input, alpha=alpha, data_format=data_format)
    # first depth wise convolution block
    # input 112 x 112 x 32
    net = depthwise_conv_block(net, 
        filters=64,
        stride=1, 
        alpha=alpha, 
        depth_multiplier=depth_multiplier, 
        block_id=1,
        data_format=data_format)
    
    # blocks with 128 output channels
    # second block
    # input 112 x 112 x 64
    net = depthwise_conv_block(net, 
        filters=128,
        stride=2, 
        alpha=alpha, 
        depth_multiplier=depth_multiplier, 
        block_id=2,
        data_format=data_format)
    # third block
    # input 56 x 56 x 128
    net = depthwise_conv_block(net, 
        filters=128,
        stride=1, 
        alpha=alpha, 
        depth_multiplier=depth_multiplier, 
        block_id=3,
        data_format=data_format)

    # blocks with 256 output channels
    # fourth block
    # input 56 x 56 x 128
    net = depthwise_conv_block(net, 
        filters=256,
        stride=2, 
        alpha=alpha, 
        depth_multiplier=depth_multiplier, 
        block_id=4,
        data_format=data_format)
    # fifth block
    # input 28 x 28 x 256
    net = depthwise_conv_block(net, 
        filters=256,
        stride=1, 
        alpha=alpha, 
        depth_multiplier=depth_multiplier, 
        block_id=5,
        data_format=data_format)

    # blocks with 512 output channels
    # sixth block
    # input 28 x 28 x 256
    net = depthwise_conv_block(net, 
        filters=512,
        stride=2, 
        alpha=alpha, 
        depth_multiplier=depth_multiplier, 
        block_id=6,
        data_format=data_format)
    # seventh to eleventh blocks (7-11) total 5 blocks
    # input 14 x 14 x 512
    for i in range(5):
        net = depthwise_conv_block(net, 
            filters=512,
            stride=1, 
            alpha=alpha, 
            depth_multiplier=depth_multiplier, 
            block_id=7+i,
            data_format=data_format)

    # blocks with 1024 output channels
    # twelfth block
    # input 14 x 14 x 512, output:  7 x 7 x 1024
    net = depthwise_conv_block(net, 
        filters=1024,
        stride=2, 
        alpha=alpha, 
        depth_multiplier=depth_multiplier, 
        block_id=12,
        data_format=data_format)
    # thirteenth block
    # input 7 x 7 x 1024 output: 7 x 7 x 1024
    # paper has a typo. it says stride=2 but it should be stride=1
    net = depthwise_conv_block(net, 
        filters=1024,
        stride=1, 
        alpha=alpha, 
        depth_multiplier=depth_multiplier, 
        block_id=13,
        data_format=data_format)

    return net


def model_mobilenet(input_tensor=None,
    input_shape=None,
    classes=1000,
    alpha=1.0,
    depth_multiplier=1,
    dropout=1e-3,
    include_top=True,
    pooling=None,
    weights=None,
    data_format="channels_last"):
    input_shape = deduce_input_shape(input_shape, require_flatten=include_top,
        weights=weights, data_format=data_format)
    if data_format == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]
    if input_tensor is not None:
        if not backend.is_keras_tensor(input_tensor):
            raise ValueError("input_tensor must be a Keras layer tensor.")
        img_input = input_tensor
    else:
        img_input =  layers.Input(shape=input_shape)
    # construct the convolutional network
    net = cnn_mobilenet(img_input, 
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        data_format=data_format)
    if include_top:
        # add the classification network
        net = top_block(net, classes=classes, alpha=alpha, dropout=dropout,
            data_format=data_format)
    else:
        # add global pooling if required
        net = pool_block(net, pooling, data_format)
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model_name = 'mobilenet_%0.2f_%s' % (alpha, rows)
    model = models.Model(inputs, net, name=model_name)
    if weights == 'imagenet':
        weight_path = get_weights_path(alpha, rows, include_top)
        model.load_weights(weight_path)
    elif weights is not None:
        model.load_weights(weights)
    return model



def get_weights_path(alpha, rows, include_top):
    if alpha == 1.0:
        alpha_text = '1_0'
    elif alpha == 0.75:
        alpha_text = '7_5'
    elif alpha == 0.50:
        alpha_text = '5_0'
    else:
        alpha_text = '2_5'
    if include_top:
        model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
    else:
        model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
    BASE_ORIGIN = ('https://github.com/fchollet/deep-learning-models/'
                        'releases/download/v0.6/')
    origin = BASE_ORIGIN + model_name
    weight_path = utils.get_file(model_name, origin,
        cache_subdir="models")
    return weight_path