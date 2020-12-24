"""
Helper functions for building convolutional neural networks
"""
import warnings

def raise_invalid_data_format(data_format):
    raise ValueError('Unknown data_format ' + str(data_format))

def check_data_format(data_format):
    if data_format not in {'channels_first', 'channels_last'}:
        raise_invalid_data_format(data_format)

def check_3d_input(input_shape):
    if len(input_shape) != 3:
        raise ValueError('Input shape must be a tuple of three integers.')

def check_3_color_channels(input_shape, data_format='channels_last'):
    n = len(input_shape) 
    if n == 2:
        raise ValueError("Input is gray-scale 2 dimensional.")
    if n != 3:
        raise ValueError("Input shape must be three dimensional.")
    index = -1 if data_format == 'channels_last' else 0
    if input_shape[index] != 3:
        raise ValueError("Input must have three color channels.")

def get_num_channels(input_shape, data_format='channels_last'):
    if len(input_shape) == 2:
        return 1
    channels = input_shape[0] if data_format == 'channels_first' else input_shape[2]
    return channels

def check_min_size(input_shape, min_size, data_format='channels_last'):
    if len(input_shape) == 2:
        hw = input_shape
    else:        
        hw = input_shape[1:] if data_format == 'channels_first' else input_shape[0:2]
    h = hw[0]
    w = hw[1] 
    if ((h is not None and h < min_size) or 
       (w is not None and w < min_size)):
       raise ValueError("Input size must be at least {}x{}, got: {}x{}".format(
        min_size, min_size, h, w
        ))

def get_channels_axis(data_format='channels_last'):
    return 3 if data_format == 'channels_last' else 1

def deduce_input_shape(input_shape=None,
    default_size=224,
    min_size=32,
    data_format='channels_last',
    require_flatten=True,
    weights=None):
    check_data_format(data_format)
    sz = default_size
    nc = 3
    if weights != 'imagenet' and input_shape:
        nc = get_num_channels(input_shape, data_format)
        if nc not in {1, 3}:
            warnings.warn('The model usually expects 1 or 3 input channels.' 
                +' However, it was passed an input_shape with {} channels.'.format(nc))
    default_shape = (nc, sz, sz) if data_format == 'channels_first' else (sz, sz, nc)

    if input_shape:
        check_3d_input(input_shape)
        check_min_size(input_shape, min_size, data_format)
        if weights == 'imagenet':
            check_3_color_channels(input_shape, data_format)
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            input_shape = (3, None, None) if data_format == 'channels_first' else (None, None, 3)
    if require_flatten and None in input_shape:
        raise ValueError("Since the network requires flattening hence, complete input shape must be specified.")
    return input_shape 
