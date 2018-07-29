'''
Example for image translation
'''
#pylint: disable=C0103

import os
import cv2
from dirsetup import IMAGES_DIR
from indigits import vision as iv

lena = os.path.join(IMAGES_DIR, 'lena.jpg')
img = cv2.imread(lena)
h, w, _  = img.shape

img2 = iv.translate(img, (20, 20))

target_size = w + 50, h + 20
img3 = iv.translate(img, (20, 20), target_size=target_size)

cv2.imshow('Lena', img)
cv2.imshow('Lena shifted', img2)
cv2.imshow('Lena shifted and enlarged', img3)


cv2.waitKey()
cv2.destroyAllWindows()
