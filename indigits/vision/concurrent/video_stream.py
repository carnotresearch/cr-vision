'''
Base class for video streams
'''
import logging
import queue
from .base import ActiveNode


class VideoReadStream(ActiveNode):
    '''A stream of video frames being read from a source'''
    frame_count = 0
    current_frame = None
    dropped_frame_count = 0

    def __init__(self, output_queue, name='VideoStream'):
        super().__init__(name)
        self.output_queue = output_queue
        self.frame_count = 0
        self.dropped_frame_count = 0

    def get_current_frame(self):
        ''' Returns current frame '''
        return self.current_frame

    def _operation(self):
        frame = self._read_next_frame()
        if frame is not None:
            self.current_frame = frame
            self.frame_count += 1
            logging.debug('Frame number: %d', self.frame_count)
            if self.output_queue is not None:
                # Put the frame in output queue if there is empty slot
                if self.output_queue.full():
                    logging.warning('Output queue is full. Frame dropped.')
                    self.dropped_frame_count += 1
                    return
                try:
                    self.output_queue.put(frame, block=False)
                except queue.Full:
                    logging.warning('Output queue is full. Frame dropped.')
                    self.dropped_frame_count += 1
                    return
        else:
            logging.warning('No frame received.')

    def _read_next_frame(self):
        ''' Returns next frame '''
        raise NotImplementedError
