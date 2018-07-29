'''
Functions for interoperability between matplot lib and opencv
'''
import matplotlib.pyplot as plt
import cv2

def imshow(image):
    '''
    Shows an opencv image using matplot lib
    '''
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(_check_colorspace(image), cmap='gray')
    plt.show()


def imshow_multiple_in_row(*args):
    '''
    Shows multiple images side by side
    '''
    num_images = len(args)
    figure = plt.figure(figsize=(16, 8))
    for i in range(num_images):
        image = args[i]
        axis = figure.add_subplot(1, num_images, i+1)
        plt.axis("off")
        axis.imshow(_check_colorspace(image), cmap='gray')


def imshow_multiple_in_grid(num_rows, *args):
    '''
    Shows multiple images in a grid
    '''
    assert isinstance(num_rows, int), 'num_rows should be an int'
    num_images = len(args)
    num_cols = (num_images + num_rows - 1) // num_rows
    figure = plt.figure(figsize=(16, 8*num_rows))
    for i in range(num_images):
        image = args[i]
        axis = figure.add_subplot(num_rows, num_cols, i+1)
        plt.axis("off")
        axis.imshow(_check_colorspace(image), cmap='gray')


def _check_colorspace(image):
    '''
    Returns same image for single channel images
    Converts from BGR to RGB color space for 3-channel images
    '''
    dtype = image.dtype
    if dtype.name not in ['uint8', cv2.CV_16U, cv2.CV_32F]:
        # Only these three dtypes are supported for color space conversion
        # We return image as it is
        return image
    if image.ndim == 3:
        _, _, channels = image.shape
        if channels == 3:
            # We assume that the image is in BGR color space
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # We assume it is a gray-scale image
    return image
