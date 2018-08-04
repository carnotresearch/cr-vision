'''
Tests the webcam sequence class
'''
#pylint: disable=C0103
import logging
import cv2
from indigits import vision as iv

webcam = iv.WebcamSequence()
for frame in webcam:
    cv2.imshow('frame', frame.frame)
    if iv.wait_for_esc_key(40):
        break
cv2.destroyAllWindows()
