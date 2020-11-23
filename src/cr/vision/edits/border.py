'''
Functions for adding borders to an image
'''

import cv2

def add_pillar_box_pattern(img, width):
    '''
    Adds a pillar box pattern to an image
    '''
    top, bottom = 0, 0
    left, right = width, width
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

def add_letter_box_pattern(img, width):
    '''
    Adds a letter box pattern to an image
    '''
    top, bottom = width, width
    left, right = 0, 0
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


def add_window_box_pattern(img, top_width, left_width):
    '''
    Adds a window box pattern to an image
    '''
    top, bottom = top_width, top_width
    left, right = left_width, left_width
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

def add_border(img, width, blue=0, green=0, red=0, color=None):
    '''
    Adds a border of a specified color 

    One can specify color in BGR format or specify each component: B, G, R separately.
    '''
    top, bottom, left, right = width, width, width, width
    if color is None:
        color = [blue, green, red]
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def add_multiple_borders(img, widths, colors):
    '''
    Adds multiple borders with specified colors and widths
    '''
    if not isinstance(widths, (list, tuple)):
        widths = [widths]
    for (i, color) in enumerate(colors):
        width = widths[i % len(widths)]
        top, bottom, left, right = width, width, width, width
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img
