'''
Adds Gaussian noise to an image
'''

import cv2
import numpy as np


class GaussianNoiseSpec:
    '''
    Specification for addition of Gaussian noise
    '''
    location = 0
    scale = 1


def add_gaussian_noise(image, spec=GaussianNoiseSpec()):
    '''
    Adds a Gaussian noise to an image.
    '''
    # Create Gaussian noise image of appropriate size with specified mean and standard deviation
    gauss = np.random.normal(spec.location, spec.scale, image.shape)
    gauss = gauss.astype(image.dtype)
    # Add the noise to the image with saturation
    noisy_image = cv2.add(image, gauss)
    # Return noisy image
    return noisy_image

