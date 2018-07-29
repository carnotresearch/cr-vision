'''
Common color conversion routines
'''
import cv2

def bgr_to_gray(image):
    '''Converts from BGR to RGB'''
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bgr_to_rgb(image):
    '''Converts from BGR to RGB

    This is useful for codes which expect images to be in RGB format
    '''
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image):
    '''Converts from RGB to BGR'''
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def bgr_to_hsv(image):
    '''Converts from BGR to HSV color space

    This is useful for filtering on the basis of specific colors
    '''
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
