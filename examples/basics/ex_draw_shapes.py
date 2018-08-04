'''
Draws simple shapes
'''
import os
import numpy as np
import cv2
from indigits import vision as iv
from dirsetup import IMAGES_DIR
#pylint: disable=C0103


image = iv.blank_image(600, 600)


top_left = (100, 100)
bottom_right = (200, 200)
color = iv.GOLD
thickness = cv2.FILLED
cv2.rectangle(image, top_left, bottom_right, color, thickness=thickness)


radius = 50
center = (300, 300)
thickness = cv2.FILLED
line_type = cv2.LINE_AA
color = iv.GREEN
# create a filled circle
cv2.circle(image, center, radius, color,
           thickness=thickness, lineType=line_type)

center = (100, 400)
radii = (100, 50)
tilt_angle = 60
start_angle = 0
end_angle = 360
color = iv.ALICEBLUE
cv2.ellipse(image, center, radii, tilt_angle, start_angle,
            end_angle, color, thickness=thickness)

points = np.array([[400, 50], [400, 300], [500, 150]], np.int32)
polygon = points.reshape((-1, 1, 2))
polygon_list = [points]
color = iv.MAGENTA
cv2.fillPoly(image, polygon_list, color)

cv2.imshow('shapes', image)
cv2.waitKey()
filepath = os.path.join(IMAGES_DIR, 'shapes.png')
cv2.imwrite(filepath, image)

