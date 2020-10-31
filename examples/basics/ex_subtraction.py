'''
Demonstrates the differences in subtraction implemented in numpy, opencv and a proper solution by promoting the types
to nearest signed type.
'''
#pylint: disable=C0103

import numpy as np
import cv2

from cr import vision as vision


a = np.arange(9, dtype=np.uint8).reshape(3, 3)
print(a)

b = np.full((3, 3), 4, np.uint8)
print(b)

c = np.subtract(b, a)
print(c)

d = cv2.subtract(b, a)
print(d)

e = vision.signed_subtract(b, a)
print(e.dtype)
print(e)

b = np.full((3, 3), 4, np.uint16)
print(b)

e = vision.signed_subtract(b, a)
print(e.dtype)
print(e)

f = vision.abs_uint8(e)
print(f)

a = np.array([2, 200, 300], dtype=np.int16)
print(vision.abs_uint8(a))
print(vision.abs_uint8(a, factor=2))
