"""Algorithms for measuring focus in an image
"""

import cv2
from cr import vision


def focus(image, format="bgr", method="laplacian_variance"):
    """Measures the level of focus in an image

    :param image: Input image
    :type image: array_like
    :param format: Format of input image,
        defaults to 'bgr'.
    :type format: str, optional
    :param method: Method to compute focus
    :type method: str, optional


    :return: Amount of focus in image
    :rtype: double

    Available methods:

    * Variance of Laplacian
    
    """
    if method == "laplacian_variance":
        return laplacian_variance(image, format)


def laplacian_variance(image, format="bgr"):
    """
    """
    image = vision.to_gray(image)
    # compute the Laplacian of the image
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # Compute the variance of Laplacian
    focus = laplacian.var()
    return focus


