'''
Utility functions for working with image data types
'''

import numpy as np

def is_integer_image(image):
    '''
    Checks if image contains integers
    '''
    return issubclass(image.dtype.type, np.integer)


def is_floating_image(image):
    '''
    Checks if image contains floating point numbers
    '''
    return issubclass(image.dtype.type, np.floating)


def is_gray_scale(image):
    '''Checks if image is gray scale'''
    if image.ndim == 2:
        return True
    if image.ndim == 3 and image.shape[2] == 1:
        return True
    return False

def is_3channel(image):
    '''Checks if image has 3 channels'''
    if image.ndim == 3 and image.shape[2] == 3:
        return True
    return False

def peak_value(image):
    '''
    Returns the peak value for a given image by its data type
    '''
    if is_integer_image(image):
        return np.iinfo(image.dtype).max
    if is_floating_image(image):
        return np.finfo(image.dtype).max
    # Unknown image data type
    return -1


def signed_subtract(image_x, image_y):
    '''
    Computes difference of images after promoting the unsigned images to the higher signed image types if required.

    No saturation is involved in this operation.
    '''
    new_type = np.result_type(image_x, image_y, np.byte)
    return (image_x.astype(new_type) - image_y.astype(new_type))


def abs_uint8(image, factor=None):
    '''
    Computes the absolute values of entries in image and maps them to UINT8
    '''
    image = abs(image)
    max_value = image.max()
    if factor is not None:
        image = (image / factor)
    image[image > 255] = 255
    return image.astype(np.uint8)
