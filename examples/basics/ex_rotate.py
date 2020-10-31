'''
Example for image rotation
'''
#pylint: disable=C0103

import os
import cv2
from dirsetup import IMAGES_DIR
from cr import vision as vision

lena = os.path.join(IMAGES_DIR, 'lena.jpg')
img = cv2.imread(lena)
h, w, _  = img.shape


angles = [10, 20, 30, 40, 50,60, 70, 80, 90]
for angle in angles:
    name = 'Lena@' + str(angle)
    cv2.namedWindow(name)
    cv2.moveWindow(name, angle*10, angle)
    rotated_image = vision.rotate(img, angle)
    cv2.imshow(name, rotated_image)

cv2.waitKey()
cv2.destroyAllWindows()
