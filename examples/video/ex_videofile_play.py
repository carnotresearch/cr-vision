'''
Shows how to play a video
'''
#pylint: disable=C0103
import os
import cv2
from cr import vision as vision
from dirsetup import VIDEO_DIR

filepath = os.path.join(VIDEO_DIR, 'nascar_01.mp4')

video_seq = vision.VideoFileSequence(filepath)
#assert video_seq.is_open()
dm = vision.DisplayManager('Nascar')
for frame in video_seq:
    dm.show(frame.frame)
    if vision.wait_for_esc_key(40):
        break
dm.stop()
