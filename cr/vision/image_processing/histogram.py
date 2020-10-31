'''
Wrapper functions for histogram calculations
'''
import numpy as np
import cv2

from cr.vision.errors import check_u8c1, check_u8c3


def histogram_u8c1(image, mask=None):
    '''Calculates histogram of a grayscale 8-bit image'''
    check_u8c1(image)
    images = [image]
    channels = [0]
    histogram_sizes = [256]
    ranges = [0, 256]
    hist = cv2.calcHist(images, channels, mask, histogram_sizes, ranges)
    return hist

def backproject_roi_u8c1(image, roi_top, roi_left, roi_width, roi_height):
    '''Computes the histogram of a region of interest and backprojects on rest of image'''
    check_u8c1(image)
    roi = image[roi_top:(roi_top+roi_height), roi_left:(roi_left+roi_width)]
    histogram_sizes = [256]
    ranges = [0, 256]
    histogram = cv2.calcHist([roi], [0], None, histogram_sizes, ranges)
    # Normalize the histogram values to the range [0, 255]
    cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
    # back project the histogram to the whole image
    ranges = [0, 256]
    scale = 1
    back_projection = cv2.calcBackProject(
        [image], [0], histogram, ranges, scale)
    return back_projection

def histogram_u8_rgb_channelwise(image, mask=None):
    '''Calculates the color histogram of 8bit RGB image separately for each channel'''
    check_u8c3(image)
    channels = cv2.split(image)
    histograms = []
    histogram_sizes = [256]
    ranges = [0, 256]
    for channel in channels:
        hist = cv2.calcHist([channel], [0], mask, histogram_sizes, ranges)
        histograms.append(hist)
    return histograms


def histogram_u8_rgb(image, mask=None, histogram_sizes=(8, 8, 8), ranges=(0, 256, 0, 256, 0, 256)):
    '''Calculates the color histogram of 8bit RGB image'''
    return histogram_u8c3(image, mask, histogram_sizes, ranges)


def histogram_u8_hsv_hs(image, mask=None, histogram_sizes=(32, 32), ranges=(0, 180, 0, 256)):
    '''Calculates the color histogram of 8 bit HSV image for H and S channels'''
    check_u8c3(image)
    images = [image]
    channels = [0, 1]
    hist = cv2.calcHist(images, channels, mask, histogram_sizes, ranges)
    return hist

def histogram_u8_hsv_hue(image, max_hue=180, min_saturation=60., min_value=32.):
    '''Computes the hue histogram after masking out not low light saturation and value ranges'''
    # mask out values where s is not in [60, 255] and v is not in [32, 255]
    mask = cv2.inRange(image, np.array((0., min_saturation, min_value)),
                      np.array((180., 255., 255.)))
    hue_histogram = cv2.calcHist([image], [0], mask, [max_hue], [0, max_hue])
    return hue_histogram

def histogram_u8c3(image, mask, histogram_sizes, ranges):
    '''Calculates the color histogram of a color 8-bit image'''
    check_u8c3(image)
    images = [image]
    channels = [0, 1, 2]
    hist = cv2.calcHist(images, channels, mask, histogram_sizes, ranges)
    return hist


def normalize_histogram_to_u8(histogram):
    '''Normalizes the histogram so that the values map to the range 0 to 255'''
    result = np.zeros(histogram.shape, dtype='uint8')
    cv2.normalize(histogram, result, 0, 255, cv2.NORM_MINMAX)
    return result
