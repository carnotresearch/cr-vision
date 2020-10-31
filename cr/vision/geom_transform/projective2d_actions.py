import cv2
import numpy as np

def translate(img, translation, target_size=None):
    '''
    Translates an image by a particular amount.

    translation: x, y

    target_size: w, h
    '''
    tx, ty = translation
    translation_matrix = np.float32([ [1,0,tx], [0,1,ty]])
    return _warp_affine(img, translation_matrix, target_size)


def rotate(img, theta):
    '''
    Rotates an image from its center. Adjust the size of the output image so that rotated
    image is fully contained inside it.

    Steps
    - Estimate the size of the bounding box for the rotated image.
    - Locate the center of the target image
    - Rotate the image about its center
    - Move its center to the target image center

    '''
    h, w = img.shape[:2]
    center = w//2, h//2
    angle = theta
    scale = 1
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # get the cosine value
    c = np.abs(rotation_matrix[0, 0])
    # get the sine value
    s = np.abs(rotation_matrix[0, 1])
    # compute the size of final image
    target_width = int((h * s) + (w * c))
    target_height = int((h * c) + (w * s))
    target_size = (target_width, target_height)
    # we need to translate the image to the new center after we rotate.
    target_center = target_width / 2,  target_height / 2
    center_translation = np.subtract(target_center, center)
    tx, ty = center_translation
    # incorporate this translation to the rotation matrix
    rotation_matrix[0, 2] += tx
    rotation_matrix[1, 2] += ty
    return _warp_affine(img, rotation_matrix, target_size)

def _warp_affine(img, warp_matrix, target_size=None):
    num_rows, num_cols = img.shape[:2]
    if target_size:
        num_cols, num_rows = target_size
    warped_image = cv2.warpAffine(img, warp_matrix, (num_cols, num_rows))
    return warped_image
