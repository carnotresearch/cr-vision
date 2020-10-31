'''
Shows how to play a video
'''
#pylint: disable=C0103
import os
import logging
import functools
import cv2
from cr import vision as vision
from dirsetup import VIDEO_DIR

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

filepath = os.path.join(VIDEO_DIR, 'nascar_01.mp4')

video_seq = vision.VideoFileSequence(filepath)
assert video_seq.is_open()

video_seq = vision.frame_sequence_map(video_seq,
                                  functools.partial(vision.resize_by_width, target_width=800))
dm = vision.DisplayManager('Nascar')
# get the first frame
first_frame = next(video_seq)
initial_bounding_box = dm.select_roi(first_frame.frame)
tracker = vision.create_object_tracker(vision.OPENCV_MIL_TRACKER)
result = tracker.initialize(first_frame.frame, initial_bounding_box)
logging.info("Tracking  initialization result: %d", result)

for current_frame in video_seq:
    if current_frame is None:
        continue
    bounding_box = tracker.update(current_frame.frame)
    if bounding_box is not None:
        (x, y, w, h) = [int(v) for v in bounding_box]
        cv2.rectangle(current_frame.frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)
    dm.show(current_frame.frame)
    key = vision.wait_for_key(40)
    if key == 27:
        break
    if key == ord('s'):
        # we will update the bounding box
        initial_bounding_box = dm.select_roi(current_frame.frame)
        tracker = vision.create_object_tracker(vision.OPENCV_MIL_TRACKER)
        result = tracker.initialize(current_frame.frame, initial_bounding_box)
        logging.info("Tracking  initialization result: %d", result)
dm.stop()
