'''
Example for adding a logo to an image
'''
#pylint: disable=C0103

import os
import cv2
from dirsetup import IMAGES_DIR
from cr import vision as vision

lena = os.path.join(IMAGES_DIR, 'lena.jpg')
lena_image = cv2.imread(lena)
fragile = os.path.join(IMAGES_DIR, 'logo_fragile.png')
logo_image = cv2.imread(fragile)
logo_image = cv2.resize(logo_image, (0, 0), fx=0.5, fy=0.5)
result = vision.add_logo(lena_image, logo_image)
cv2.imshow('Lena with logo', result)

cv2.waitKey()
cv2.destroyAllWindows()
