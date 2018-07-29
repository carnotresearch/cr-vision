'''
Methods for filtering

We cover implementations for common use cases. One can fall back to the
low level OpenCV API for more flexibility if needed.
'''
import cv2

from indigits.vision.errors import check_ndim, check_nchannels

def filter_2d(image, kernel):
    '''
    Filters an image retaining its depth
    '''
    return cv2.filter2D(image, -1, kernel)


def gaussian_blur(image, kernel_size=5, sigma=0):
    '''
    Filters an image using Gaussian filter of a given kernel size where we
    assume that the filter is symmetric.
    '''
    kernel_size = (kernel_size, kernel_size)
    return cv2.GaussianBlur(image, kernel_size, sigma)


def laplacian(image, ddepth=cv2.CV_32F):
    '''Computes Laplacian of gray scale images '''
    check_ndim(image.ndim, 2, 3)
    if image.ndim == 3:
        _, _, channels = image.shape
        check_nchannels(1, channels)
    return cv2.Laplacian(image, ddepth)


def sobel_x(image, ddepth=cv2.CV_32F, ksize=3):
    '''Computes sobel derivative in x direction'''
    return cv2.Sobel(image, ddepth, 1, 0, ksize=ksize)


def sobel_y(image, ddepth=cv2.CV_32F, ksize=3):
    '''Computes sobel derivative in y direction'''
    return cv2.Sobel(image, ddepth, 0, 1, ksize=ksize)
