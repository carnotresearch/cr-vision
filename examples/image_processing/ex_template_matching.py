'''
Showing the functionality of template matching
'''
#pylint: disable=C0103
import os
import numpy as np
import cv2
from cr import vision as vision
from dirsetup import IMAGES_DIR

names = [
    'alarm_clock.jpg',
    'barbara.png',
    'fighter_jet.jpg',
    'pug.jpg',
    'stuff.jpg'
]
dm = vision.DisplayManager(['Image', 'Template'], gap_x=800)
for name in names:
    # path of image to read
    image_path = os.path.join(IMAGES_DIR, name)
    assert os.path.exists(image_path)
    # read the image
    image = cv2.imread(image_path)
    # select the bounding box for template image
    bounding_box = dm.select_roi(image)
    # unpack the bounding box
    x_1, y_1, width, height = bounding_box
    # check if user didn't specify bounding box
    if width == 0 or height == 0:
        # we didn't select a bounding box.
        break
    # extract the template from image
    template = image[y_1:(y_1 + height), x_1:(x_1 + width)]
    # choose a random smaller width for the template
    target_width = np.random.randint(10, width+1)
    # now resize the template to this random size preserving aspect ratio
    template = vision.resize_by_width(template, target_width)
    # match the template
    result = vision.match_template_multscale(image, template)
    # unpack the matching details
    (max_value, rectangle, scale) = result
    # draw the matched rectangle
    cv2.rectangle(image, *rectangle, color=vision.GREEN, thickness=2)
    # show the template and its match inside image
    dm.show_all(image, template)
    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        break
