'''
An example for creating a border around an image
'''
#pylint: disable=C0103

import os
import cv2
from dirsetup import IMAGES_DIR
from cr import vision as vision
from cr.vision.image_processing import colors

lena_path = os.path.join(IMAGES_DIR, 'lena.jpg')
img = cv2.imread(lena_path)
img2 = vision.add_border(img, width=20, color=colors.ROYALBLUE)
name = 'Lena with border'
cv2.namedWindow(name)
cv2.moveWindow(name, 10, 10)
cv2.imshow(name, img2)


img3 = vision.add_multiple_borders(img, widths=20, colors=[
    colors.BLUE, colors.RED, colors.GREEN])
name = 'Lena with multiple borders'
cv2.namedWindow(name)
cv2.moveWindow(name, 40, 20)
cv2.imshow(name, img3)

cv2.waitKey()
cv2.destroyAllWindows()
