'''
Webcam frame sequence
'''
import os
import errno
import cv2

from .interfaces import FrameSequence
from ..video.frame import Frame

class VideoCaptureSequence(FrameSequence):
    '''Wrapper for cv2.VideoCapture'''

    stream = None
    '''OpenCV Video Capture Stream'''

    def __init__(self):
        self.frame_number = 0
        # Let's capture the time when frame sequence was created
        self.tick = cv2.getTickCount()

    def __next__(self):
        '''Returns next frame (along with its frame number)'''
        if self.stream is None:
            raise StopIteration
        grabbed, frame = self.stream.read()
        if grabbed is True:
            self.frame_number += 1
            return Frame(frame, self.frame_number)
        else:
            raise StopIteration


    def __iter__(self):
        return self

    def is_done(self):
        return self.stream is None

    def is_open(self):
        if self.stream is None:
            return False
        return self.stream.isOpened()

    def stop(self):
        '''Stop serving more frames'''
        if self.stream is None:
            # nothing to do
            return
        self.stream.release()
        self.stream = None

    def __del__(self):
        #  Ensure cleanup
        self.stop()


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

class WebcamSequence(VideoCaptureSequence):
    '''A frame sequence from webcamp'''

    def __init__(self):
        self.stream = cv2.VideoCapture(0)
        super().__init__()


class VideoFileSequence(VideoCaptureSequence):
    '''Frame sequence from a video file'''

    def __init__(self, filepath):
        if not os.path.exists(filepath):
            raise IOError(errno.ENOENT, "Missing file", filepath)
        if  os.path.isdir(filepath):
            raise IOError(errno.EISDIR, 'is directory', filepath)
        self.filepath = filepath
        self.stream = cv2.VideoCapture(filepath)
        if self.stream is None:
            raise IOError(errno.EIO, 'does not appear to be a supported video file', filepath)
        super().__init__()
