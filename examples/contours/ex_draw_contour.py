import numpy as np
import cv2
from cr import vision
from dirsetup import IMAGES_DIR



def get_one_contour():
    """Returns a 'fixed' contour"""

    cnts = [np.array(
        [[[600, 320]], [[563, 460]], [[460, 562]], [[320, 600]], [[180, 563]], [[78, 460]], [[40, 320]], [[77, 180]],
         [[179, 78]], [[319, 40]], [[459, 77]], [[562, 179]]], dtype=np.int32)]
    return cnts

cnts = get_one_contour()
print(cnts)


image_path =IMAGES_DIR / 'girl.png'

print (image_path)
image = cv2.imread(str(image_path))
cv2.drawContours(image, cnts, -1, (0,255,0), 3)
cv2.imshow("Contours", image)


key = cv2.waitKey(0) & 0xFF
