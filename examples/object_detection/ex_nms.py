'''
Showing the functionality of non maximum suppression
'''
#pylint: disable=C0103
import numpy as np
import cv2
from cr import vision as vision
from cr.vision import object_detection as od

bounding_boxes = np.array([
    (12, 84, 140, 210),
    (24, 84, 152, 222),
    (36, 84, 160, 212),
    (14, 96, 140, 224),
    (24, 96, 152, 224),
    (24, 104, 142, 236)])

image1 = vision.blank_image(400, 400)
image2 = image1.copy()
dm = vision.DisplayManager(['all boxes', 'after suppression'], gap_x=400)
for bbox in bounding_boxes:
    x_0, y_0, x_1, y_1 = bbox
    cv2.rectangle(image1, (x_0, y_0), (x_1, y_1), vision.RED, 2)
dm.show(image1, 0)
final_bb = od.non_maximum_suppression(bounding_boxes)
print(od.bb_areas(bounding_boxes))
print(od.bb_areas(final_bb))
for bbox in final_bb:
    x_0, y_0, x_1, y_1 = bbox
    cv2.rectangle(image2, (x_0, y_0), (x_1, y_1), vision.RED, 2)
dm.show(image2, 1)
cv2.waitKey(0)
