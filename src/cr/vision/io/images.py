import numpy as np
import imageio

from cr.vision.core.scaling import resize_crop


def read_images(paths, target_width, target_height):
    images = []
    for path in paths:
        image = imageio.imread(path)
        image = resize_crop(image, target_width, target_height)
        images.append(image)
    return np.array(images)