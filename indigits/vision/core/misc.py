'''
Simple miscellaneous operations
'''
import numpy as np
import cv2



def blank_image(width, height, channels=3, dtype='uint8'):
    '''Constructs a blank image'''
    return np.zeros((width, height, channels), dtype=dtype)

def single_color_image(width, height, color):
    '''Constructs a color image of a given color'''
    color  = np.array(color, dtype='uint8')
    print(type(color))
    image = np.tile(color, width*height)
    return image.reshape(height, width, color.size)

def keep_masked_values(image, mask):
    '''
    Keeps values in image for which mask is non-zero
    '''
    return cv2.bitwise_and(image, image, mask=mask)

def discard_masked_values(image, mask):
    '''
    Discards values in image for which mask is non-zero
    '''
    return cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))


def to_abs_u8(image):
    '''Computes absoulte values and maps to gray scale 8 bit'''
    # Compute absolute values
    image = np.abs(image)
    # get max value
    max_value = image.max()
    if max_value < 128 or max_value > 255:
        # Map them between 0 to 255
        image *= (255.0/image.max())
    # map them to 8 bit unsigned
    return image.astype('uint8')
