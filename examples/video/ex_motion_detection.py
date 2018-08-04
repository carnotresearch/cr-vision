'''
Tests the webcam stream class
'''
#pylint: disable=C0103
import queue
import logging
from indigits import vision as iv

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

webcam_frame_queue = queue.Queue(maxsize=20)
md_output_queue = queue.Queue(maxsize=20)
webcam_stream = iv.WebcamReadStream(webcam_frame_queue)
motion_detector = iv.SimpleMotionDetectionNode(
    webcam_frame_queue, md_output_queue)
osd = iv.OnScreenDisplay(md_output_queue)
graph = iv.ActiveGraph([webcam_stream, motion_detector, osd])
graph.start()
osd.finish_event.wait()
graph.stop()
