'''
Visual effects on images


References:

* https://github.com/jmhobbs/libcvfx
* https://github.com/PacktPublishing/OpenCV-3-x-with-Python-By-Example
* https://docs.gimp.org/en/plug-in-convmatrix.html
* https://stackoverflow.com/questions/22654770/creating-vignette-filter-in-opencv

'''
import numpy as np
import cv2
from indigits import vision as iv


def mirror_lr(image):
    '''
    Mirrors an image between left to right

    Args:
        image (array): Input image

    Returns:
        Mirrored image
    '''
    return np.fliplr(image)


def mirror_ud(image):
    '''
    Mirrors an image between up and down

    Args:
        image (array): Input image

    Returns:
        Mirrored image
    '''
    return np.flipud(image)


def mirror_x(image):
    '''
    Mirrors the image diagonally.
        Top left corner goes to bottom rigth and vice versa. 
        Top right corner goes to bottom left and vice versa.

    Args:
        image (array): Input image

    Returns:
        Mirrored image
    '''
    image = np.fliplr(image)
    image = np.flipud(image)
    return image


def select_red(image, others=0):
    '''
    Selects only the red channel of the image

    Args:
        image (array): Input image
        others (int): Value to be set in other channels (default 0)

    Returns:
        Reddish image
    '''
    image = image.copy()
    image[:, :, 0] = others
    image[:, :, 1] = others
    return image


def select_green(image, others=0):
    '''
    Selects only the green channel of the image

    Args:
        image (array): Input image
        others (int): Value to be set in other channels (default 0)

    Returns:
        Greenish image
    '''
    image = image.copy()
    image[:, :, 0] = others
    image[:, :, 2] = others
    return image


def select_blue(image, others=0):
    '''
    Selects only the blue channel of the image

    Args:
        image (array): Input image
        others (int): Value to be set in other channels (default 0)

    Returns:
        Bluish image
    '''
    image = image.copy()
    image[:, :, 1] = others
    image[:, :, 2] = others
    return image


def monochrome(image):
    '''
    Converts image to gray level

    Args:
        image (array): Input image

    Returns:
        Monochrome image
    '''
    return iv.bgr_to_gray(image)


EMBOSS_FILTERS = {
    'default': np.array([[-2, -1, 0],
                         [-1, 1, 1],
                         [0, 1, 2]
                         ]) /3,
    'SW': np.array([[0, -1, -1],
                    [1, 0, -1],
                    [1, 1, 0]]),
    'NE': np.array([[-1, -1, 0],
                    [-1, 0, 1],
                    [0, 1, 1]]),
    'NW': np.array([[1, 0, 0],
                    [0, 0, 0],
                    [0, 0, -1]])
}


def emboss(image, direction='default'):
    '''
    Creates an embossed version of an image in a given direction

    Args:
        image (array): Input image

    Returns:
        Embossed image
    '''
    if direction != 'default':
        image = iv.bgr_to_gray(image)
    kernel = EMBOSS_FILTERS[direction]
    image = cv2.filter2D(image, -1, kernel) + 128
    return image


def motion_blur(image, kernel_size=3, horz=True):
    '''
    Introduces a motion blur effect

    Args:
        image (array): Input image

    Returns:
        Motion blurred image
    '''
    kernel = np.zeros((kernel_size, kernel_size))
    # make middle row all ones
    mid_row = int((kernel_size - 1)/2)
    kernel[mid_row, :] = np.ones(kernel_size) / kernel_size
    if not horz:
        # for vertical blurring, we need to get the transpose
        kernel = kernel.T
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image


def sharpen(image):
    '''Sharpens an image using edge enhancement

    Args:
        image (array): Input image

    Returns:
        Sharpened image
    '''
    kernel = np.array([[-1, -1, -1, -1, -1],
                       [-1, 2, 2, 2, -1],
                       [-1, 2, 8, 2, -1],
                       [-1, 2, 2, 2, -1],
                       [-1, -1, -1, -1, -1]]) / 8.0
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


def vignette(image, vignette_size=100):
    '''
    Applies vignette effect on an image

    Args:
        image (array): Input image

    Returns:
        Vignetted image
    '''
    # height and width of the image
    height, width = image.shape[:2]
    # We will construct the vignette mask using Gaussian kernels
    # Construct a Gaussian kernel for x-direction
    kernel_x = cv2.getGaussianKernel(width, vignette_size)
    # Construct a Gaussian kernel for y-direction
    kernel_y = cv2.getGaussianKernel(height, vignette_size)
    # Multiply them together to form a combined Gaussian kernel
    kernel = kernel_x*kernel_y.T
    # compute the norm of the Gaussian kernel
    kernel_norm = np.linalg.norm(kernel)
    # prepare the mask
    mask = 255 * kernel / kernel_norm
    if image.ndim == 2:
        # it's a black and white images
        output = image*mask
    else:
        channels = image.shape[2]
        if channels == 1:
            # it's a black and white image
            output = image*mask
        else:
            # it's a BGR image
            b,g,r = cv2.split(image)
            b = b*mask
            g = g*mask
            r = r*mask
            output = cv2.merge(b,g,r)
    return output


def vignette(image, vignette_size=150):
    '''
    Applies vignette effect on an image

    Args:
        image (array): Input image

    Returns:
        Vignetted image
    '''
    # height and width of the image
    height, width = image.shape[:2]
    # We will construct the vignette mask using Gaussian kernels
    # Construct a Gaussian kernel for x-direction
    kernel_x = cv2.getGaussianKernel(width, vignette_size)
    # Construct a Gaussian kernel for y-direction
    kernel_y = cv2.getGaussianKernel(height, vignette_size)
    # Multiply them together to form a combined Gaussian kernel
    kernel = kernel_y*kernel_x.T
    # prepare the mask
    mask = kernel / kernel.max()
    if image.ndim == 2:
        # it's a black and white images
        output = image*mask
    else:
        channels = image.shape[2]
        if channels == 1:
            # it's a black and white image
            output = image*mask
        else:
            # it's a BGR image
            b, g, r = cv2.split(image)
            b = b*mask
            g = g*mask
            r = r*mask
            output = cv2.merge([b, g, r])
    return output.astype('uint8')
