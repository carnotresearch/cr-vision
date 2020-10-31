'''
Functions to add a logo to an image
'''
import cv2

from .placement import place_image_at


def add_logo(image, logo_image, x_pos=0, y_pos=0, mask_threshold=10):
    '''
    Adds a logo to an image at the specified position. 

    Assumes that the background in logo image is black.
    '''
    rows, cols, channels = logo_image.shape
    # let's construct a gray-level version of the logo image
    logo_gray = cv2.cvtColor(logo_image, cv2.COLOR_BGR2GRAY)
    # Let's identify the non-zero parts of logo image
    max_value = 255
    _, logo_mask = cv2.threshold(logo_gray, mask_threshold,
                                 max_value, cv2.THRESH_BINARY)
    # Let's create a copy of the original image
    target_image = image.copy()
    place_image_at(target_image, logo_image, logo_mask, x_pos, y_pos)
    return target_image
