'''
Displays images on screen
'''
import logging
import queue
import threading
import numpy as np
import cv2
from cr import vision as iv
from .base import ActiveNode


class OnScreenDisplay(ActiveNode):
    '''OnScreenDisplay'''

    def __init__(self, input_queue, name='Screen'):
        super().__init__(name)
        self.input_queue = input_queue
        self.finish_event = threading.Event()

    def _operation(self):
        if self.finish_event.is_set():
            # nothing to do
            return
        # check if user has asked for the display to be stopped.
        if iv.wait_for_esc_key():
            # We indicate that display should be stopped now
            self.finish_event.set()
        try:
            # read next set of frames
            token = self.input_queue.get(timeout=1)
            if isinstance(token, np.ndarray):
                # a single frame is to be shown
                cv2.imshow('Output', token)
        except queue.Empty:
            # there were no frames in the input queue
            logging.debug('on screen display: input queue is empty')
            pass

    def _end(self):
        super()._end()
        cv2.destroyAllWindows()
