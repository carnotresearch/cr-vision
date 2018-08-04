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
assert video_seq.is_open()


def resize_to_width(frame_sequence):
    '''Resizes frames in a sequence to width 500'''
    for cur_frame in frame_sequence:
        if cur_frame is None:
            yield cur_frame
            continue
        cur_frame.frame = iv.resize_by_width(cur_frame.frame, 800)
        yield cur_frame


video_seq = resize_to_width(video_seq)

dm = iv.DisplayManager('Nascar')
tracker = iv.create_object_tracker(iv.OPENCV_MIL_TRACKER)
# get the first frame
first_frame = next(video_seq)
initial_bounding_box = dm.select_roi(first_frame.frame)
tracker.initialize(first_frame.frame, initial_bounding_box)

for frame in video_seq:
    if frame is None:
        continue
    bounding_box = tracker.update(frame.frame)
    if bounding_box is not None:
        (x, y, w, h) = [int(v) for v in bounding_box]
        cv2.rectangle(frame.frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)
    dm.show(frame.frame)
    if iv.wait_for_esc_key(40):
        break
dm.stop()
