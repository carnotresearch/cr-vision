import cv2
from cr import vision
from dirsetup import IMAGES_DIR

image_path =IMAGES_DIR / 'boy.bmp'

image = cv2.imread(image_path)
