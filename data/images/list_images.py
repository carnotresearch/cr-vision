import os
import sys
import cv2

curdir = os.path.dirname(__file__)
files = os.listdir(curdir)
for file in files:
    _, ext = os.path.splitext(file)
    if ext == '.py': 
        continue
    img = cv2.imread(file)
    size = img.shape
    h, w  = size[:2]
    print('{} {}x{}'.format(file, w, h))