import os
import cv2
from dirsetup import IMAGES_DIR

lena = os.path.join(IMAGES_DIR, 'lena.jpg')
img = cv2.imread(lena)
h, w, _  = img.shape

from cr.vision.core.scaling import resize, resize_crop

img = resize(img, 200, 400)

center = resize_crop(img, 300, 300)

print(img.shape)
print(center.shape)

cv2.imshow('Lena', img)
cv2.imshow('Lena cropped', center)

cv2.waitKey()
cv2.destroyAllWindows()
