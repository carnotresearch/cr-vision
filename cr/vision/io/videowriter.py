'''
Wrapper for OpenCV video writer
'''

import cv2


class VideoWriter:
    '''Wrapper class for OpenCV video writer'''

    def __init__(self, filepath, fourcc='XVID', fps=15, frame_size=(640, 480), is_color=True):
        '''Constructor'''
        self.filepath = filepath
        if isinstance(fourcc, str):
            fourcc = cv2.VideoWriter_fourcc(*fourcc)
        elif isinstance(fourcc, int):
            pass
        else:
            raise "Invalid fourcc code"
        self.stream = cv2.VideoWriter(filepath, fourcc, fps, frame_size)

    def write(self, frame):
        '''Writes a frame to output file'''
        self.stream.write(frame)

    def is_open(self):
        '''Returns if the stream is open for writing'''
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
