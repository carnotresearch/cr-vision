

import cv2
import numpy as np
from cr import vision
from cr.vision import io
from dirsetup import IMAGES_DIR
from cr.vision import object_detection as od

image_path =IMAGES_DIR / 'harajuku_pedestrians_on_omotesando_04_15739934765.jpg'

print (image_path)
image = cv2.imread(str(image_path))


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = vision.resize(image, target_width=min(400, image.shape[1]))
orig = image.copy()
# detect people in the image
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
    padding=(8, 8), scale=1.05)

initial_bb = np.array([(x, y, x + w, y + h) for (x, y, w, h) in rects])

final_bb = od.non_maximum_suppression(initial_bb)

# draw the original bounding boxes
for (x1, y1, x2, y2) in final_bb:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
dm = io.DisplayManager(['Original', 'People'], gap_x=600)

dm.show_all(orig, image)

key = cv2.waitKey(0) & 0xFF

