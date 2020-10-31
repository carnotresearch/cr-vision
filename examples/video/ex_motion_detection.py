'''
Tests the webcam stream class
'''
#pylint: disable=C0103
import queue
import logging
from cr import vision as vision

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

webcam_frame_queue = queue.Queue(maxsize=20)
md_output_queue = queue.Queue(maxsize=20)
webcam_stream = vision.WebcamReadStream(webcam_frame_queue)
motion_detector = vision.SimpleMotionDetectionNode(
    webcam_frame_queue, md_output_queue)
osd = vision.OnScreenDisplay(md_output_queue)
graph = vision.ActiveGraph([webcam_stream, motion_detector, osd])
graph.start()
osd.finish_event.wait()
graph.stop()
