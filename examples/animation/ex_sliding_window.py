import cv2
import numpy as np
import imageio
from cr import vision
from dirsetup import IMAGES_DIR

image_path =IMAGES_DIR / 'pool.png'

print (image_path)
image = cv2.imread(str(image_path))
size = image.shape[0:2]
height = size[0]
width = size[1]
print ("size: width: {}, height: {}".format(width, height))
# Window height
h = 128
# Window width
w = 128

sw = 32
sh = 32

nh = int( (height + sh - h)/ sh)
nw = int( (width + sw  - w)  / sw)

x1, y1, x2, y2 = 0, 0, w, h
stop = False


writer = imageio.get_writer('animation.mp4', fps=10)

for r in range(nh):
    x1 = 0
    x2 = w
    y1 += sh
    y2 += sh
    for c in range(nw):
        print("r: {}, c: {}".format(r+1, c+1))
        working = image.copy()
        cv2.rectangle(working, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('Sliding Windows', working)
        key = cv2.waitKey(100) & 0xFF
        # we need to flip the color space before writing
        working = vision.bgr_to_rgb(working)
        writer.append_data(working)
        if key == ord('q') or key == 27:
            stop = True
            break
        x1 += sw
        x2 += sw
    if stop: break
writer.close()
key = cv2.waitKey(0) & 0xFF
