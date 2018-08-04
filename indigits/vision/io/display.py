'''
Display of frames on screen
'''
import cv2
from indigits import vision as iv

class DisplayManager:
    '''
    Manages the display of one or more OpenCV 
    windows
    '''
    def __init__(self, window_names):
        if isinstance(window_names, str):
            window_names = [window_names]
        self.window_names = window_names
        for (i, window_name) in enumerate(self.window_names):
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name, i*10, i*10)

    def show(self, frame, window_num=0):
        '''Shows a frame on a specified window'''
        window_name = self.window_names[window_num]
        cv2.imshow(window_name, frame)

    def select_roi(self, frame, window_num=0):
        '''Selects a region of interest in a given frame'''
        window_name = self.window_names[window_num]
        bounding_box = cv2.selectROI(window_name, frame, fromCenter=False,
                       showCrosshair=True)
        return bounding_box

    def stop(self):
        '''Stops the display of all windows'''
        cv2.destroyAllWindows()

    def __del__(self):
        self.stop()
