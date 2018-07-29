'''
Example for image translation
'''
#pylint: disable=C0103

import os
import cv2
from dirsetup import IMAGES_DIR
from indigits import vision as iv

image_path = os.path.join(IMAGES_DIR, 'bookpage_dark_scan.jpg')
image = cv2.imread(image_path)
h, w, _ = image.shape

gray_image = iv.bgr_to_gray(image)
thresholded_image = iv.adaptive_threshold_gaussian(gray_image, block_size=115, constant=1)
cv2.imshow('Original', image)
cv2.moveWindow('Original', 10, 10)
cv2.imshow('Corrected', thresholded_image)
cv2.waitKey()
cv2.destroyAllWindows()
