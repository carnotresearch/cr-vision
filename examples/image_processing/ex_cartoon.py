import os
import cv2
from cr import vision
from cr.vision import io
from dirsetup import IMAGES_DIR

image_path = os.path.join(IMAGES_DIR, 'puppy.jpg')

puppy = cv2.imread(image_path)

modified = vision.cartoonize(puppy)


dm = io.DisplayManager(['Image', 'Cartoon'], gap_x=800)

dm.show_all(puppy, modified)

key = cv2.waitKey(0) & 0xFF
