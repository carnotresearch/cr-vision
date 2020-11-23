'''
Wrapper class and methods for OpenCV saliency module
'''

import cv2
from cr import vision as iv

class Saliency:
    '''Wrapper class for saliency models'''

    def __init__(self, saliency):
        self.saliency = saliency

    def compute_saliency(self, image):
        '''Computes the saliency map'''
        result, saliency_map = self.saliency.computeSaliency(image)
        if result:
            return saliency_map
        return None

    def compute_saliency_uint8(self, image):
        '''Computes the saliency map and returns as 8-bit integers'''
        result, saliency_map = self.saliency.computeSaliency(image)
        if result:
            return (saliency_map*255).astype('uint8')
        return None

    def compute_saliency_mask(self, image):
        '''Computes the saliency mask using Otsu binarization'''
        result, saliency_map = self.saliency.computeSaliency(image)
        if result:
            # perform otsu binarization
            saliency_map = iv.threshold_otsu(saliency_map)
            return saliency_map
        return None

#pylint: disable=C0103,E1101
def create_static_saliency_fine_grained():
    '''Creates a static saliency estimator based on fine grained method'''
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    return Saliency(saliency)

def create_static_saliency_spectral_residual():
    '''Creates a static saliency estimator based on spectral residual method'''
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    return Saliency(saliency)
