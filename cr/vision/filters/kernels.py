'''
Standard Kernels
'''

import numpy as np

KERNEL_BOX_3X3 = np.ones((3, 3), dtype="float") * (1.0 / (3*3))
KERNEL_BOX_5X5 = np.ones((5, 5), dtype="float") * (1.0 / (5*5))
KERNEL_BOX_7X7 = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
KERNEL_BOX_21X21 = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))


KERNEL_SHARPEN_3X3 = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

KERNEL_LAPLACIAN_3X3 = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
KERNEL_SOBEL_X = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
KERNEL_SOBEL_Y = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")
