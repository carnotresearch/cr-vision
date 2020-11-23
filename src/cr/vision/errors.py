'''
Exception classes for cr-vision library
'''

# Definitive guide to Python exceptions https://julien.danjou.info/python-exceptions-guide/


class CRVError(Exception):
    '''Base exception class'''


class InvalidNumDimensionsError(CRVError):
    '''Invalid number of dimensions error'''

class InvalidNumChannelsError(CRVError):
    ''' Invalid number of channels error'''

    def __init__(self, expected_channels, actual_channels):
        message = 'Invalid number of channels. Expected: {}, Actual: {}'.format(
            expected_channels, actual_channels)
        super().__init__(message)


class NotU8C1Error(CRVError):
    '''Image is not grayscale 8 bit unsigned'''


class NotU8C3Error(CRVError):
    '''Image is not 8 bit unsigned 3 channel color image'''


class UnsupportedImageFormatError(CRVError):
    """Unsupported image format"""

def check_ndim(actual_ndim, expected_min_ndim=None, expected_max_ndim=None):
    ''' Checks if the number of dimensions is correct'''
    message = None
    if expected_min_ndim is not None and expected_max_ndim is not None:
        if expected_min_ndim == expected_max_ndim:
            if actual_ndim != expected_min_ndim:
                message = 'Invalid number of dimensions. Expected: {}, Actual: {}'.format(
                    expected_min_ndim, actual_ndim)
        else:
            if actual_ndim < expected_min_ndim or actual_ndim > expected_max_ndim:
                message = 'Invalid dimensions. Expected between: {}-{}, Actual: {}'.format(
                    expected_min_ndim, expected_max_ndim, actual_ndim)
    elif expected_min_ndim is not None:
        if actual_ndim < expected_min_ndim:
            message = 'Expected Minimum: {}, Actual: {}'.format(
                expected_min_ndim, actual_ndim)
    elif expected_max_ndim is not None:
        if actual_ndim > expected_max_ndim:
            message = 'Expected Maximum: {}, Actual: {}'.format(
                expected_max_ndim, actual_ndim)
    if message is not None:
        raise InvalidNumDimensionsError(message)


def check_nchannels(expected_channels, actual_channels):
    '''Checks if number of channels is correct'''
    if actual_channels != expected_channels:
        raise InvalidNumChannelsError(expected_channels, actual_channels)

def check_u8c1(image):
    '''Checks that the image is an unsigned 8 bit image with one channel'''
    if image.dtype != 'uint8':
        raise NotU8C1Error('The image data type is not unsigned 8 bit')
    if image.ndim == 1:
        raise NotU8C1Error('It is a vector. Expected an image.')
    elif image.ndim == 2:
        # all good
        pass
    elif image.ndim == 3:
        if image.shape[2] != 1:
            raise NotU8C1Error('Image has more than one channels')
    else:
        raise NotU8C1Error('Invalid dimensions')


def check_u8c3(image):
    '''Chcecks that the image is an unsigned 8 bit image with 3 channels'''
    if image.dtype != 'uint8':
        raise NotU8C3Error('The image data type is not unsigned 8 bit')
    if image.ndim != 3:
        raise NotU8C3Error('Image must have 3 dimensions')
    if image.shape[2] != 3:
        raise NotU8C3Error('Image must have 3 channels')



def error_unsupported_image_format(format=None):
    """Raises UnsupportedImageFormatError exception"""
    if format is None:
        raise UnsupportedImageFormatError("Unsupported image format")
    message = "Unsupported image format: {}".format(format)
    raise UnsupportedImageFormatError(message)