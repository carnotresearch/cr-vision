import os
import cv2
from cr import vision as vision
from dirsetup import VIDEO_DIR
import skvideo.io
import rx
from rx import operators as ops
import asyncio

loop = asyncio.get_event_loop()
done = loop.create_future()

filepath = os.path.join(VIDEO_DIR, 'nascar_01.mp4')

inputparameters = {}
outputparameters = {}

videogen = skvideo.io.vreader(filepath, 
                inputdict=inputparameters,
                outputdict=outputparameters)


source = rx.from_iterable(videogen).pipe(
    ops.map(vision.rgb_to_bgr))
dm = vision.DisplayManager('Nascar')


class Player:
    def __init__(self):
        self.count = 0

    def on_next(self, frame):
        self.count += 1
        print("frame: {} {}".format(self.count, frame.shape))
        dm.show(frame)
        vision.wait_for_esc_key(40)
        pass

    def on_error(self, error):
        pass

    def on_completed(self):
        print("completed")
        done.set_result(0)



subscription = source.subscribe(Player())
loop.run_until_complete(done)
loop.close()
subscription.dispose()

