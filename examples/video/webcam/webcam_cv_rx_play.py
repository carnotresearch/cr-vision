import asyncio
import cv2
import rx
from rx import operators as ops
from cr import vision
from cr.vision.crx import EventLoopPlayer


# Web cam
webcam = vision.WebcamSequence()

# Event loop
loop = asyncio.get_event_loop()

# On Screen Playback
player = EventLoopPlayer(loop)

# Frame rate
fps = 10
# clock
clock_source = rx.interval(1/fps)


with webcam:
    composed = clock_source.pipe(
        ops.map(lambda _: next(webcam).frame),
        # ops.map(lambda frame: cv2.resize(frame,(256,224))),
    )
    composed.subscribe(player)
    loop.run_forever()

loop.close()
