'''
Tests the webcam sequence class
'''
#pylint: disable=C0103
import logging
import cv2
from cr import vision as vision

webcam = vision.WebcamSequence()
for frame in webcam:
    cv2.imshow('frame', frame.frame)
    if vision.wait_for_esc_key(40):
        break
cv2.destroyAllWindows()
