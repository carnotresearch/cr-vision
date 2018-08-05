'''
Shows how to create single color images
'''
import cv2
from indigits import vision as iv

image = iv.single_color_image(500, 500, iv.GOLD)
cv2.imshow('image', image)
cv2.waitKey()

image = iv.single_color_image(500, 500, 128)
cv2.imshow('image', image)
cv2.waitKey()

