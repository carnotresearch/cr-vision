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


def frame_sequence_map(input_sequence, mapping):
    '''Maps a sequence of frames to another sequence after applying a mapping'''
    for cur_frame in input_sequence:
        if cur_frame is None:
            yield cur_frame
            continue
        cur_frame.frame = mapping(cur_frame.frame)
        yield cur_frame
