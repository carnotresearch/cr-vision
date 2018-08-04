'''
Shows how to play a video
'''
#pylint: disable=C0103
import os
import cv2
from indigits import vision as iv
from dirsetup import VIDEO_DIR

filepath = os.path.join(VIDEO_DIR, 'nascar_01.mp4')

video_seq = iv.VideoFileSequence(filepath)
#assert video_seq.is_open()
dm = iv.DisplayManager('Nascar')
for frame in video_seq:
    dm.show(frame.frame)
    if iv.wait_for_esc_key(40):
        break
dm.stop()
