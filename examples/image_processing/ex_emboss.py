import os
import cv2
from cr import vision
from cr.vision.core import effects
from cr.vision import io

from dirsetup import IMAGES_DIR

image_path = os.path.join(IMAGES_DIR, 'puppy.jpg')

puppy = cv2.imread(image_path)

a = effects.emboss(puppy, format="rgb")

b = effects.emboss(puppy, format="bgr")


dm = io.DisplayManager(['Image', 'Emboss (RGB)', 'Emboss (BGR)'], gap_x=200)

dm.show_all(puppy, a, b)

key = cv2.waitKey(0) & 0xFF

