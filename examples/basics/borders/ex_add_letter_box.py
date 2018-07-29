'''
An example for creating a letter box around an image
'''
#pylint: disable=C0103

import os
import cv2
from dirsetup import IMAGES_DIR
from indigits import vision as iv

lena_path = os.path.join(IMAGES_DIR, 'lena.jpg')
img = cv2.imread(lena_path)

img = iv.add_letter_box_pattern(img, 100)
name = 'Lena with letter box'
cv2.namedWindow(name)
cv2.moveWindow(name, 10, 10)
cv2.imshow(name, img)
cv2.waitKey()
cv2.destroyAllWindows()
