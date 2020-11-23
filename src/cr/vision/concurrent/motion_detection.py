'''
Detects motion in video
'''
import logging
import queue
from cr import vision as iv

from .base import ActiveNode



class SimpleMotionDetectionNode(ActiveNode):
    '''Detects motion using a simple algorithm'''

    first_frame = None

    def __init__(self, input_queue, output_queue, name='SimpleMotionDetectionAgent'):
        super().__init__(name)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.detector = iv.SimpleMotionDetector()

    def _begin(self):
        # wait till the first frame is obtained
        first_frame = None
        while first_frame is None:
            try:
                first_frame = self.input_queue.get(timeout=1)
                self.detector.begin(first_frame)
            except queue.Empty:
                logging.warning('Waiting for first frame')

    def _operation(self):
        try:
            frame = self.input_queue.get(timeout=1)
            result = self.detector.process(frame)
            self.output_queue.put(result)
        except queue.Empty:
            pass

    def _end(self):
        super()._end()
        self.detector.end()
