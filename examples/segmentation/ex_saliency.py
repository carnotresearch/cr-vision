#pylint: disable=C0103
import os
import logging
import cv2
from indigits import vision as iv
from dirsetup import IMAGES_DIR

names = [
    'alarm_clock.jpg',
    'barbara.png',
    'fighter_jet.jpg',
    'pug.jpg',
    'stuff.jpg'
]

dm = iv.DisplayManager(['Image', 'Saliency'], gap_x=800)

saliency = iv.create_static_saliency_fine_grained()
for name in names:
    image_path = os.path.join(IMAGES_DIR, name)
    assert os.path.exists(image_path)
    image = cv2.imread(image_path)
    saliency_mask = saliency.compute_saliency_mask(image)
    dm.show_all(image, saliency_mask)
    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        break
