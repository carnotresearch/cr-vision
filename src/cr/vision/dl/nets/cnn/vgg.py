"""
VGG-16 and VGG-19 networks

References

* `How to Develop VGG, Inception and ResNet Modules from Scratch in Keras <https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/>`_

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend, models, utils

from .utils import deduce_input_shape
from cr.vision.dl.nets.publicweights import FCHOLLET

def vgg_block(input_layer, n_filters, n_conv_layers, name):
    """
    Creates a VGG-Block
    """
    cur_layer = input_layer
    # Add convolutional layers 
    for i in range(n_conv_layers):
        layer_name='{}_conv{}'.format(name, i+1)
        cur_layer = layers.Conv2D(n_filters, (3,3), 
            padding='same', activation='relu', name=layer_name)(cur_layer)
    # Add max pooling layer
    layer_name='{}_pool'.format(name)
    cur_layer = layers.MaxPooling2D( (2,2), strides=(2,2), name=layer_name)(cur_layer)
    # Return the block
    return cur_layer


def cnn_vgg16(img_input):
    # block 1
    net = vgg_block(img_input, 64, 2, 'block1')
    # block 2
    net = vgg_block(net, 128, 2, 'block2')
    # block 3
    net = vgg_block(net, 256, 3, 'block3')
    # block 4
    net = vgg_block(net, 512, 3, 'block4')
    # block 5
    net = vgg_block(net, 512, 3, 'block5')    
    return net

def cnn_vgg19(img_input):
    # block 1
    net = vgg_block(img_input, 64, 2, 'block1')
    # block 2
    net = vgg_block(net, 128, 2, 'block2')
    # block 3
    net = vgg_block(net, 256, 4, 'block3')
    # block 4
    net = vgg_block(net, 512, 4, 'block4')
    # block 5
    net = vgg_block(net, 512, 4, 'block5')    
    return net


def top_vgg(cnn, classes=1024):
    # Classification block
    x = layers.Flatten(name='flatten')(cnn)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    return x


def pool_cnn(cnn, pooling=None):
    if pooling == 'avg':
        cnn = layers.GlobalAveragePooling2D()(cnn)
    elif pooling == 'max':
        cnn = layers.GlobalMaxPooling2D()(cnn)
    return cnn


def model_vgg16(input_tensor=None, 
    input_shape=None, 
    classes=1000,
    include_top=True,
    pooling=None,
    weights=None):
    # provide a way to compute the default input shape
    input_shape = deduce_input_shape(input_shape, require_flatten=include_top,
        weights=weights)
    if input_tensor is not None:
        if not backend.is_keras_tensor(input_tensor):
            raise ValueError("input_tensor must be a Keras layer tensor.")
        img_input = input_tensor
    else:
        img_input =  layers.Input(shape=input_shape)
    net = cnn_vgg16(img_input)
    if include_top:
        # add the top classification network
        net = top_vgg(net, classes)
    else:
        # add global pooling if required
        net = pool_cnn(net, pooling)
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, net, name='vgg16')
    if weights == 'imagenet':
        resource = FCHOLLET['VGG16']
        resource = resource['WITH_TOP'] if include_top else resource['NO_TOP']
        origin = resource.uri
        resource_path = utils.get_file(resource.name, origin,
            cache_subdir="models", file_hash=resource.file_hash)
        model.load_weights(resource_path)
    elif weights is not None:
        model.load_weights(weights)
    return model


def model_vgg19(input_tensor=None, 
    input_shape=None, 
    classes=1000,
    include_top=True,
    pooling=None,
    weights=None):
    # provide a way to compute the default input shape
    input_shape = deduce_input_shape(input_shape, require_flatten=include_top,
        weights=weights)
    if input_tensor is not None:
        if not backend.is_keras_tensor(input_tensor):
            raise ValueError("input_tensor must be a Keras layer tensor.")
        img_input = input_tensor
    else:
        img_input =  layers.Input(shape=input_shape)
    net = cnn_vgg19(img_input)
    if include_top:
        # add the top classification network
        net = top_vgg(net, classes)
    else:
        # add global pooling if required
        net = pool_cnn(net, pooling)
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, net, name='vgg19')
    if weights == 'imagenet':
        resource = FCHOLLET['VGG19']
        resource = resource['WITH_TOP'] if include_top else resource['NO_TOP']
        origin = resource.uri
        resource_path = utils.get_file(resource.name, origin,
            cache_subdir="models", file_hash=resource.file_hash)
        model.load_weights(resource_path)
    elif weights is not None:
        model.load_weights(weights)
    return model
