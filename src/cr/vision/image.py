"""
Basic manipulations on an NP array representing an image


References

* `A Gentle Introduction to Channels-First and Channels-Last Image Formats <https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/>`_


Tensorflow

* Channels are last.

Theano

* Channels are first.

CNTK

* Channels are last.


CR-VISION Recommended Settings

* Channels are last
* BGR
* Epsilon : 1e-07
* Float data type : 'float32'
* Deep learning backend : Tensorflow



"""
import numpy as np

FLOATX = 'float32'


def _raise_invalid_data_format(data_format):
    raise ValueError('Unknown data_format ' + str(data_format))

def expand_gray_channels_first(image):
    """
    Expands a gray-scale image from (h, w) to (1, h, w) format
    """
    assert image.ndim == 2
    image = np.expand_dims(image, axis=0)
    return image

def expand_gray_channels_last(image):
    """
    Expands a gray-scale image from (h, w) to (h, w, 1) format
    """
    assert image.ndim == 2
    image = np.expand_dims(image, axis=2)
    return image


def expand_gray(image, data_format="channels_last"):
    assert image.ndim == 2
    if data_format == 'channels_last':
        return np.expand_dims(image, axis=2)
    elif data_format == 'channels_first':
        return np.expand_dims(image, axis=0)
    else:
        _raise_invalid_data_format(data_format)


def expand_to_batch(image):
    """
    Expands an image to a batch of images (of batch size 1)
    """
    batch = np.expand_dims(image, axis=0)
    return batch


def channels_last_to_first(image):
    """
    Moves channels of a color image from last to first dimension
    """
    assert image.ndim == 3
    image = np.moveaxis(image, 2, 0)
    return image


def channels_first_to_last(image):
    """
    Moves channels of a color image from first to last dimension
    """
    assert image.ndim == 3
    image = np.moveaxis(image, 0, 2)
    return image



def normalize_1(image):
    """
    Normalize image to (-1, 1) range
    """
    if not issubclass(image.dtype.type, np.floating):
        image = image.astype(FLOATX, copy=False)
    image /= 127.5
    image -= 1.
    return image

def normalize_0_1(image):
    """
    Normalize image to 0-1 range
    """
    if not issubclass(image.dtype.type, np.floating):
        image = image.astype(FLOATX, copy=False)
    image /= 255.
    return image


def subtract_mean(image, mean, data_format='channels_last'):
    if data_format == 'channels_last':
            image[..., 0] -= mean[0]
            image[..., 1] -= mean[1]
            image[..., 2] -= mean[2]
    elif data_format == 'channels_first':
        if image.ndim == 3:
            image[0, :, :] -= mean[0]
            image[1, :, :] -= mean[1]
            image[2, :, :] -= mean[2]
        elif x.ndim == 4:
            image[:, 0, :, :] -= mean[0]
            image[:, 1, :, :] -= mean[1]
            image[:, 2, :, :] -= mean[2]
        else:
            raise ValueError('Unsupported number of dimensions.')
    else:
        _raise_invalid_data_format(data_format)
    return image

def scale_down(image, std, data_format='channels_last'):
    if data_format == 'channels_last':
        image[..., 0] /= std[0]
        image[..., 1] /= std[1]
        image[..., 2] /= std[2]
    elif data_format == 'channels_first':
        if image.ndim == 3:
            image[0, :, :] /= std[0]
            image[1, :, :] /= std[1]
            image[2, :, :] /= std[2]
        elif image.ndim == 4:
            image[:, 0, :, :] /= std[0]
            image[:, 1, :, :] /= std[1]
            image[:, 2, :, :] /= std[2]
        else:
            raise ValueError('Unsupported number of dimensions.')
    else:
        _raise_invalid_data_format(data_format)
    return image

def normalize_imagenet(x, data_format="channels_last", color_format="rgb", mean=True, scale=True):
    """
    Subtract the image-net mean and scale by image-net standard deviation.
    We assume that image is in RGB format.
    """
    if not issubclass(x.dtype.type, np.floating):
        # Make sure that image is in (0,1) range
        x = normalize_0_1(x)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if color_format == 'bgr':
        # We need to change the order
        mean = mean[::-1]
        std = std[::-1]
    if mean:
        x = subtract_mean(x, mean)
    if scale:
        x = scale_down(x, std)
    return x

def normalize_torch(image, data_format="channels_last"):
    image = normalize_0_1(image)
    image = normalize_imagenet(image, data_format)
    return image


def swap_rgb_channels(image, data_format="channels_last"):
    if data_format == 'channels_first':
        if image.ndim == 3:
            # RGB to BGR swap on axis 0
            image = image[::-1, ...]
        else:
            # RGB to BGR swap on axis 1
            image = image[:, ::-1, ...]
    else:
        # RGB to BGR swap on last axis
        image = image[..., ::-1]
    return image

def preprocess_caffe(image, data_format="channels_last", color_format="rgb"):
    """
    Perform following transformations:
    - Convert from RGB to BGR if needed
    - Subtract the Image Net mean to zero center it
    """
    if not issubclass(image.dtype.type, np.floating):
        # Make sure that image is in (0,1) range
        image = image.astype(FLOATX, copy=False)
    if color_format == 'rgb':
        image = swap_rgb_channels(image, data_format)
    # mean in BGR format
    mean = [103.939, 116.779, 123.68]
    image = subtract_mean(image, mean, data_format)
    return image


########################################################
#
# Helper functions for working with segmentation masks
#
#########################################################

def binary_mask_to_rgba(mask, color="red"):
    """
    Converts a binary segmentation mask to a specific color
    Also adds an alpha channel
    """
    # height and width
    h, w = mask.shape[0:2]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    if color == 'red':
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    if color == 'green':
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    if color == 'blue':
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    if color == 'yellow':
        return np.stack((ones, ones, zeros, ones), axis=-1)
    if color == 'magenta':
        return np.stack((ones, zeros, ones, ones), axis=-1)
    if color == 'cyan':
        return np.stack((zeros, ones, ones, ones), axis=-1)
    if color == 'white':
        return np.stack((ones, ones, ones, ones), axis=-1)
    assert "Unsupported color"
