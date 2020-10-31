'''
Shows how to create single color images
'''
import cv2
from cr import vision as vision

image = vision.single_color_image(500, 500, vision.GOLD)
cv2.imshow('image', image)
cv2.waitKey()

image = vision.single_color_image(500, 500, 128)
cv2.imshow('image', image)
cv2.waitKey()

