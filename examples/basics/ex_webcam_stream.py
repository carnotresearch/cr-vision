'''
Tests the webcam stream class
'''
#pylint: disable=C0103
import queue
import logging
from cr.vision import WebcamReadStream
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
output_queue = queue.Queue()
stream = WebcamReadStream(output_queue)
stream.start()
for _ in range(20):
    output_queue.get()
stream.stop()
print(output_queue.qsize())
