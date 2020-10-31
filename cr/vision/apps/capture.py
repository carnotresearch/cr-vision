'''
A driver routine to capture data from video camera.
'''
import cv2
# import numpy as np


def webcam_capture(title='Camera', processor=None):
    '''
    Captures and displays frames from video camera.
    '''
    cap = cv2.VideoCapture(0)
    while True:   
        _, frame = cap.read()
        cv2.imshow(title, frame)
        if processor is not None:
            frame = processor(frame)
        k = cv2.waitKey(40) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    webcam_capture()
