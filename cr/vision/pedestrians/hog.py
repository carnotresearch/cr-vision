import cv2
from cr import vision



class HOGDetector:
    def __init__(self,
        win_stride=(4,4),
        padding=(8,8),
        scale=1.05):
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.detector = hog

    def __call__(self, image):
        hog = self.detector
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, 
            winStride=self.win_stride,
            padding=self.padding, 
            scale=self.scale)
        return (rects, weights)
