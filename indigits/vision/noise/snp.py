'''
Adds Salt and Pepper noise to an image
'''

import cv2
import numpy as np

from indigits.vision import core

class SaltAndPepperNoiseSpec:
    '''
    Specification for addition of Salt and Pepper noise to an image
    '''
    copy = True
    '''By default destination image will be a copy of source image and source image will be left untouched'''

    black_threshold = .02
    '''Values less than 2% of peak value will be considered as locations for pepper noise'''

    white_threshold = .98
    '''Values greater than 98% of peak value will be considered as locations for salt noise'''


def add_salt_and_pepper_noise(image, spec=SaltAndPepperNoiseSpec()):
    '''
    Adds salt and pepper noise to a given image
    '''
    rows, cols, _ = image.shape
    # Let's get the maximum value for the image data type
    peak_value = core.peak_value(image)
    pattern = np.random.randint(
        low=0, high=peak_value, size=(rows, cols), dtype=image.dtype)
    pepper_pattern = pattern < peak_value * spec.black_threshold
    salt_pattern = pattern > peak_value * spec.white_threshold
    out = image.copy()
    out[pepper_pattern] = 0
    out[salt_pattern] = peak_value
    return out

# def noisy(noise_typ,image)
    
#     elif noise_typ == "s&p":
#     elif noise_typ == "poisson":
#         vals = len(np.unique(image))
#         vals = 2 ** np.ceil(np.log2(vals))
#         noisy = np.random.poisson(image * vals) / float(vals)
#         return noisy
#     elif noise_typ =="speckle":
#         row,col,ch = image.shape
#         gauss = np.random.randn(row,col,ch)
#         gauss = gauss.reshape(row,col,ch)        
#         noisy = image + image * gauss
#         return noisy
# share
