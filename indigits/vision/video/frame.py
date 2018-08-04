'''
Frames in a video sequence
'''
import cv2

class Frame:
    '''A frame object'''
    
    frame = None
    '''Current frame'''

    _frame_number = 0
    '''The number of frame in the sequence'''
    
    _tick = 0
    ''' The clock tick when the frame object was created'''
    
    def __init__(self, frame, frame_number):
        self.frame = frame
        self._frame_number = frame_number
        self._tick = cv2.getTickCount()

    @property
    def frame_number(self):
        '''Returns underlying frame number'''
        return self._frame_number

    @property
    def tick(self):
        '''Returns the clock tick when the frame was created'''
        return self._tick
