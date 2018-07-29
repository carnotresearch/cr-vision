'''
Basic operations on images
'''

import cv2

def element_wise_eq(image_1, image_2):
    '''Computes  1 == 2 for each element'''
    return cv2.compare(image_1, image_2, cv2.CMP_EQ)


def element_wise_gt(image_1, image_2):
    '''Computes  1 > 2 for each element'''
    return cv2.compare(image_1, image_2, cv2.CMP_GT)

def element_wise_ge(image_1, image_2):
    '''Computes  1 >= 2 for each element'''
    return cv2.compare(image_1, image_2, cv2.CMP_GE)

def element_wise_lt(image_1, image_2):
    '''Computes  1 < 2 for each element'''
    return cv2.compare(image_1, image_2, cv2.CMP_LT)

def element_wise_le(image_1, image_2):
    '''Computes  1 <= 2 for each element'''
    return cv2.compare(image_1, image_2, cv2.CMP_LE)

def element_wise_ne(image_1, image_2):
    '''Computes  1 != 2 for each element'''
    return cv2.compare(image_1, image_2, cv2.CMP_NE)
