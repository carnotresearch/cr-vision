import numpy as np
import imageio

from cr.vision.core.scaling import resize_crop
from cr.vision.core.cvt_color import gray_to_rgb

def is_gray(image):
    return (image.ndim == 2) or (image.ndim == 3 and image.shape[2] == 1)

def read_images(paths, target_width, target_height):
    images = []
    for path in paths:
        image = imageio.imread(path)
        if is_gray(image):
            print(f'{path}, {image.shape}')
            # make sure that image is rgb (and not gray scale)
            image = gray_to_rgb(image)
            print(f'shape after conversion {image.shape}')
        image = resize_crop(image, target_width, target_height)
        images.append(image)
    return np.array(images)