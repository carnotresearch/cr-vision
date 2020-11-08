'''
Implementation of seam carving method

References:

    * `Seam Carving Python Module by Vivian Hylee <https://github.com/vivianhylee/seam-carving/blob/master/seam_carving.py>`_
    * `Source Code for OpenCV 3.x with Python By Example, Chapter 6 <https://github.com/PacktPublishing/OpenCV-3-x-with-Python-By-Example/tree/master/Chapter06>`_
    * `2017, Adrian Roserbrock, Seam carving with OpenCV, Python, and scikit-image  <https://www.pyimagesearch.com/2017/01/23/seam-carving-with-opencv-python-and-scikit-image/>`_
    * `scikit-image transform module <http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.seam_carve>`_
    * `scikit-image Seam Carving <http://scikit-image.org/docs/dev/auto_examples/transform/plot_seam_carving.html>`_

'''
import logging
import numpy as np
from skimage import transform, img_as_float
from cr import vision


def seam_carve(image, target_width, target_height, energy_map=vision.sobel_energy_l1):
    '''Converts an image into target width and height by using seam carving

    :param image: Input image to be reduced to target width and height
    :type image: array_like
    :param target_width: target width of image
    :type target_width: int
    :param target_height: target height of image
    :type target_height: int
    :param energy_map: A function used to construct the energy matrix for the image
    :type energy_map: function

    :return: Seam carved image in which vertical and horizontal seams
        have been identified and removed to meet the target image size.
    :rtype: ndarray

    '''
    src_height, src_width = image.shape[:2]
    t_height = int(float(src_height) * target_width / src_width)
    if t_height > target_height:
        logging.info('Reducing image size to %d x %d', target_width, t_height)
        # Let's do a first level resizing
        image = vision.resize_by_width(image, target_width)
        src_height, src_width = image.shape[:2]
    t_width = int(float(src_width) * target_height / src_height)
    if t_width > target_width:
        logging.info('Reducing image size to %d x %d', t_width, target_height)
        image = vision.resize_by_height(image, target_height)
        src_height, src_width = image.shape[:2]
    if target_width < src_width:
        # compute the energy matrix for the image
        energy_matrix = energy_map(image)
        # We need to remove some vertical seams
        num_seams_to_remove = src_width - target_width
        logging.info('Removing %d vertical seams', num_seams_to_remove)
        image = transform.seam_carve(image, energy_matrix,
                                     'vertical', num_seams_to_remove)
        image = (image*255).astype('uint8')
    if target_height < src_height:
        # compute the energy matrix for the image
        energy_matrix = energy_map(image)
        # We need to remove some horizontal seams
        num_seams_to_remove = src_height - target_height
        logging.info('Removing %d horizontal seams', num_seams_to_remove)
        image = transform.seam_carve(image, energy_matrix,
                                     'horizontal', num_seams_to_remove)
        image = (image*255).astype('uint8')
    return image


def find_vertical_seam(vseam_matrix):
    '''Finds the indices of the vertical seam with minimum energy

    :param vseam_matrix: matrix of vertical seams
    :type vseam_matrix: array-2d

    :return: Indices of the vertical seam with minimum energy
    :rtype: uint32 array
    '''
    # the number of rows and columns in the cumulative energy matrix
    rows, cols = vseam_matrix.shape
    # The vector of indices of seam of minimum energy
    output = np.zeros((rows,), dtype=np.uint32)
    # Find the pixel with minimum energy in last row
    output[-1] = np.argmin(vseam_matrix[-1])
    # iterate over remaining rows
    for row in range(rows-2, -1, -1):
        # we will find the minimum energy of the neighboring three pixels
        # in previous row
        # we also need to take care of the boundary conditions
        # get the value of pixel index for previous row
        previous_index = output[row+1]
        # check if we are at the left boundary
        if previous_index == 0:
            # we just need to consider columns 0 and 1
            output[row] = np.argmin(vseam_matrix[row, :2])
        else:
            # let's consider if the previous index is on the right boundary
            last_index = min(previous_index+2, cols)
            # pick the neighbors in the 2 or 3 positions
            neighbors = vseam_matrix[row, (previous_index-1):last_index]
            output[row] = np.argmin(neighbors) + previous_index - 1
    return output


def delete_vertical_seam(image, seam):
    '''Deletes a vertical seam from the image

    :param image: Input image
    :type image: array_like

    :param seam: Indices of the vertical seam to be deleted
    :type seam: uint32 array-1d

    :return: Image with the vertical seam deleted
    :rtype: ndarray
    '''
    rows, cols = image.shape[:2]
    if image.ndim == 2:
        out_image = np.zeros((rows, cols-1), dtype=image.dtype)
    else:
        out_image = np.zeros((rows, cols-1, image.shape[2]), dtype=image.dtype)
    for row in range(rows):
        # identify the column of pixel to be deleted in this row
        col = seam[row]
        # copy the remaining pixels
        out_image[row, :] = np.delete(image[row, :], (col), axis=0)
    return out_image


def compute_vertical_seams(energy_matrix):
    '''Computes the vertical seams by dynamic programming

    .. todo::

        * Optimize inner loop 
    '''
    n_rows, n_cols = energy_matrix.shape[:2]
    vertical_seams = energy_matrix.copy()
    for row in range(1, n_rows):
        for col in range(n_cols):
            if col == 0:
                vertical_seams[row, col] += np.min(vertical_seams[row-1, 0:2])
            elif col == n_cols - 1:
                vertical_seams[row,
                               col] += np.min(vertical_seams[row-1, n_cols-2:n_cols])
            else:
                vertical_seams[row,
                               col] += np.min(vertical_seams[row-1, col-1:col+2])
    return vertical_seams


def compute_horizontal_seams(energy_matrix):
    '''Computes the vertical seams by dynamic programming'''
    # We reuse the existing implementation of computation of vertical seams
    seams = compute_vertical_seams(energy_matrix.T)
    # We transpose the result to make them horizontal seams
    return seams.T


def find_horizontal_seam(hseam_matrix):
    '''Finds the indices of the horizontal seam with minimum energy
    '''
    # Reuse implementation
    return find_vertical_seam(hseam_matrix.T)


def delete_horizontal_seam(image, seam):
    '''Deletes a horizonal seam from the image

    :param image: Input image
    :type image: array_like

    :param seam: Indices of the horizontal seam to be deleted
    :type seam: uint32 array-1d

    :return: Image with the horizontal seam deleted
    :rtype: ndarray
    '''
    n_rows, n_cols = image.shape[:2]
    if image.ndim == 2:
        out_image = np.zeros((n_rows-1, n_cols), dtype=image.dtype)
    else:
        out_image = np.zeros(
            (n_rows-1, n_cols, image.shape[2]), dtype=image.dtype)
    for n_col in range(n_cols):
        # identify the row number of pixel to be deleted in this column
        n_row = seam[n_col]
        # copy the remaining pixels
        out_image[:, n_col] = np.delete(image[:, n_col], (n_row))
    return out_image


def reduce_width_by_seam_carving(image, target_width, energy_map):
    '''Reduces the width of an image via seam carving'''
    src_width = image.shape[1]
    delta = src_width - target_width
    for _ in range(delta):
        energy_matrix = energy_map(image)
        vertical_seams = compute_vertical_seams(energy_matrix)
        seam = find_vertical_seam(vertical_seams)
        image = delete_vertical_seam(image, seam)
    return image


def reduce_height_by_seam_carving(image, target_height, energy_map):
    '''Reduce the height of an image via seam carving'''
    src_height = image.shape[0]
    delta = src_height - target_height
    for _ in range(delta):
        energy_matrix = energy_map(image)
        horizontal_seams = compute_horizontal_seams(energy_matrix)
        seam = find_horizontal_seam(horizontal_seams)
        image = delete_horizontal_seam(image, seam)
    return image


