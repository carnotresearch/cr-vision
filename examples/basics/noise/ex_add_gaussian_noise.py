'''
Example showing how to add Gaussian noise to an image
'''
#pylint: disable=C0103
import os
import cv2
from dirsetup import IMAGES_DIR
from cr import vision as vision

lena_path = os.path.join(IMAGES_DIR, 'lena.jpg')
img = cv2.imread(lena_path)

img = vision.add_gaussian_noise(img)
name = 'Lena with Gaussian Noise'
cv2.namedWindow(name)
cv2.moveWindow(name, 40, 20)
cv2.imshow(name, img)
cv2.waitKey()
cv2.destroyAllWindows()
