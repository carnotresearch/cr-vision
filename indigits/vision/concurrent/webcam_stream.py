'''
A video stream for reading frames from webcam
'''
from .video_stream import  VideoReadStream
import cv2
import logging

class WebcamReadStream(VideoReadStream):
    '''Webcam stream'''
    
    def __init__(self, output_queue, name='WebcamStream'):
        super().__init__(output_queue, name)
        self.stream = cv2.VideoCapture(0)

    def _read_next_frame(self):
        ''' Returns next frame '''
        grabbed, frame = self.stream.read()
        if grabbed is True:
            return frame
        else:
            return None

    def _end(self):
        super()._end()
        self.stream.release()
        logging.info('The webcam device has been released.')

