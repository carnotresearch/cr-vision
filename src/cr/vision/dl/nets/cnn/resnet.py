"""

References

* https://github.com/marcopeix/Deep_Learning_AI/blob/master/4.Convolutional%20Neural%20Networks/2.Deep%20Convolutional%20Models/Residual%20Networks.ipynb

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend, models, utils

from .utils import deduce_input_shape
from cr.vision.dl.nets.publicweights import FCHOLLET, KERAS_TEAM


# epsilon for Batch Normalization layers
BN_EPS= 1.001e-5


def conv_input(img_input, preact=False, name="conv", ch_axis=3):
    """
    Convolution and Max Pooling to go from 224x224 to 56x56
    """
    # pad in advance for valid convolution (output is 230x230)
    net = layers.ZeroPadding2D(padding=(3, 3), name=name + '_pad')(img_input)
    # perform the big 7x7 convolution with 2x2 stride to (112x112x64)
    net = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name=name + '_conv')(net)
    if preact is False:
        # batch normalization before activation (output is 112x112x64)
        net = layers.BatchNormalization(axis=ch_axis, epsilon=BN_EPS, name=name + '_bn')(net)
        # relu activation (output is 112x112x64)
        net = layers.Activation('relu', name=name + "_relu")(net)
    # pad again for max pooling (output is 114x114x64)
    net = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(net)
    # 3x3 max pooling with 2x2 stride (output is 56x56x64)
    net = layers.MaxPooling2D((3, 3), strides=(2, 2), name="pool1_pool")(net)
    return net



def residual_block_131(input, filters, kernel_size=3, stride=1, name='block', ch_axis=3):
    """
    A block of three convolutional layers with an identity connection.
    There are three convolutional layers

    - 1x1
    - 3x3
    - 1x1

    The input is added to the output of the final layer before
    activation

    """
    # shape of input tensor
    input_shape = input.shape
    # number of channels in input tensor
    num_input_channels = input_shape[ch_axis]
    # number of channels in output tensor
    num_output_channels = 4 * filters
    # if input and output channels are same then we can feed
    # the input directly as identity shortcut
    # otherwise, we need to add a convolutional layer in identity path
    conv_in_identity_path = num_output_channels != num_input_channels
    if conv_in_identity_path is True:
        # add a conv layer to increase the number of channels
        shortcut = layers.Conv2D(num_output_channels, 1, strides=stride,
                                 name=name + '_0_conv')(input)
        # batch normalize (activation will come later)
        shortcut = layers.BatchNormalization(axis=ch_axis, epsilon=BN_EPS,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = input

    # The 1x1 first convolution layer on input
    # This layer may need to scale down at times so stride is provided 
    # Since it's a 1x1 layer, hence no padding is required
    net = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(input)
    net = layers.BatchNormalization(axis=ch_axis, epsilon=BN_EPS,
                                  name=name + '_1_bn')(net)
    net = layers.Activation('relu', name=name + '_1_relu')(net)

    # The 3x3 second convolution layer
    # we need to ensure that the width and height don't change. Hence padding=same.
    net = layers.Conv2D(filters, kernel_size, padding='same',
                      name=name + '_2_conv')(net)
    net = layers.BatchNormalization(axis=ch_axis, epsilon=BN_EPS,
                                  name=name + '_2_bn')(net)
    net = layers.Activation('relu', name=name + '_2_relu')(net)

    # The 1x1 third convolution layer
    net = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(net)
    net = layers.BatchNormalization(axis=ch_axis, epsilon=BN_EPS,
                                  name=name + '_3_bn')(net)

    net = layers.Add(name=name + '_add')([shortcut, net])
    net = layers.Activation('relu', name=name + '_out')(net)
    return net


def residual_stack(input, filters, blocks, stride1=2, name='stack', ch_axis=3):
    """A set of stacked residual blocks.
    """
    net = residual_block_131(input, filters, stride=stride1, name=name + '_block1', ch_axis=ch_axis)
    for i in range(2, blocks + 1):
        net = residual_block_131(net, filters, name=name + '_block' + str(i))
    return net


def cnn_resnet50(img_input, ch_axis=3):
    net = conv_input(img_input, preact=False, name="conv1", ch_axis=ch_axis)
    net = residual_stack(net, 64, 3, stride1=1, name='conv2')
    net = residual_stack(net, 128, 4, name='conv3')
    net = residual_stack(net, 256, 6, name='conv4')
    net = residual_stack(net, 512, 3, name='conv5')
    return net


def top_resnet(net, classes):
    # add the top classification network
    net = layers.GlobalAveragePooling2D(name='avg_pool')(net)
    net = layers.Dense(classes, activation='softmax', name='probs')(net)
    return net

def pool_resnet(net, pooling, data_format):
    if pooling == 'avg':
        net = layers.GlobalAveragePooling2D(name='avg_pool')(net)
    elif pooling == 'max':
        net = layers.GlobalMaxPooling2D(name='max_pool')(net)
    return net

def model_resnet50(input_tensor=None,
    input_shape=None,
    classes=1000,
    include_top=True,
    pooling=None,
    weights=None,
    data_format="channels_last"):
    ch_axis = 3 if data_format == 'channels_last' else 1
    input_shape = deduce_input_shape(input_shape, require_flatten=include_top,
        weights=weights, data_format=data_format)
    if input_tensor is not None:
        if not backend.is_keras_tensor(input_tensor):
            raise ValueError("input_tensor must be a Keras layer tensor.")
        img_input = input_tensor
    else:
        img_input =  layers.Input(shape=input_shape)
    net = cnn_resnet50(img_input, ch_axis=ch_axis)
    if include_top:
        net = top_resnet(net, classes)
    else:
        # add global pooling if required
        net = pool_resnet(net, pooling, data_format)
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, net, name='resnet50')
    if weights == 'imagenet':
        resource = KERAS_TEAM['RESNET50']
        resource = resource['WITH_TOP'] if include_top else resource['NO_TOP']
        origin = resource.uri
        resource_path = utils.get_file(resource.name, origin,
            cache_subdir="models", file_hash=resource.file_hash)
        model.load_weights(resource_path)
        pass
    elif weights is not None:
        model.load_weights(weights)
    return model
