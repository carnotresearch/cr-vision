'''
Detects motion in video
'''
import cv2
from cr import vision as iv




class SimpleMotionDetector:
    '''Detects motion using a simple algorithm'''

    first_frame = None
    current_frame = None
    gray_frame = None

    def __init__(self, minimum_contour_area=500):
        self.counter = 0
        self.minimum_contour_area = minimum_contour_area


    def _process_frame(self, frame):
        self.current_frame = iv.resize_by_width(frame, target_width=500)
        gray = iv.bgr_to_gray(self.current_frame)
        gray = iv.gaussian_blur(gray, kernel_size=21)
        self.gray_frame = gray

    def begin(self, first_frame):
        '''Handles first frame'''
        self._process_frame(first_frame)
        self.first_frame = self.gray_frame 

    def process(self, frame):
        '''Handles subsequent frames'''
        self.counter += 1
        self._process_frame(frame)
        frame_delta = cv2.absdiff(self.first_frame, self.gray_frame)
        thresholded_image = iv.threshold_above(frame_delta, 25)
        thresholded_image = cv2.dilate(thresholded_image, None, iterations=2)
        contours = iv.find_external_contours(thresholded_image)
        contours = [contour for contour in contours if contour.area()
                    > self.minimum_contour_area]
        contours = iv.Contours(contours)
        contours.draw_simple_bounding_boxes(self.current_frame, color=iv.GREEN)
        if self.counter == 20:
            self.first_frame = self.gray_frame
            self.counter = 0
        return self.current_frame

    def end(self):
        '''Cleanup'''
        pass
