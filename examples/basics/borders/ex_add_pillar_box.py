'''
An example for creating a pillar box around an image
'''
#pylint: disable=C0103

import os
import cv2
from dirsetup import IMAGES_DIR
from cr import vision
from cr.vision import edits

lena_path = os.path.join(IMAGES_DIR, 'lena.jpg')
img = cv2.imread(lena_path)

img = edits.add_pillar_box_pattern(img, 100)
cv2.imshow('Lena with pillar box', img)
cv2.waitKey()
cv2.destroyAllWindows()
