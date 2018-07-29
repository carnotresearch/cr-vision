'''
Wrappers for thresholding functions
'''

import cv2
from .types import peak_value

def threshold_above(image, threshold, mask_value=None):
    ''' Sets all values above the threshold to the mask value and 0 to remaining values
    '''
    if mask_value is None:
        # Determine the max value on the basis of data type of image
        mask_value = peak_value(image)
    _, result = cv2.threshold(image, threshold, mask_value, cv2.THRESH_BINARY)
    return result


def threshold_below(image, threshold, mask_value=None):
    ''' Sets all values below the threshold to the mask value and 0 to remaining values
    '''
    if mask_value is None:
        # Determine the max value on the basis of data type of image
        mask_value = peak_value(image)
    _, result = cv2.threshold(
        image, threshold, mask_value, cv2.THRESH_BINARY_INV)
    return result

def threshold_truncate(image, threshold):
    ''' Truncates all pixels in the image above threshold to the threshold value
    '''
    max_value  = peak_value(image)
    _, result = cv2.threshold(image, threshold, max_value, cv2.THRESH_TRUNC)
    return result

def adaptive_threshold_mean(image, mask_value=None, block_size=8, constant=0):
    '''
    Performs adaptive thresholding based on mean method.
    '''
    if mask_value is None:
        # Determine the max value on the basis of data type of image
        mask_value = peak_value(image)
    method = cv2.ADAPTIVE_THRESH_MEAN_C
    return cv2.adaptiveThreshold(image, mask_value, method, cv2.THRESH_BINARY, block_size, constant)


def adaptive_threshold_gaussian(image, mask_value=None, block_size=8, constant=0):
    '''
    Performs adaptive thresholding based on Gaussian method.
    '''
    if mask_value is None:
        # Determine the max value on the basis of data type of image
        mask_value = peak_value(image)
    method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    return cv2.adaptiveThreshold(image, mask_value, method, cv2.THRESH_BINARY, block_size, constant)


