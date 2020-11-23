# standard library
import asyncio
import click
import traceback
from dataclasses import dataclass

# Vision related
import cv2
import numpy as np
from cr import vision
from cr.vision import io
from cr.vision import filters

# Pipeline building
import rx
from rx import operators as ops
from cr.vision.crx import step
from cr.vision.crx import EventLoopPlayer

@dataclass
class Context:
    index: int = 0
    image : np.ndarray = None
    gray : np.ndarray = None


class MotionDetector:

    def __init__(self, minimum_contour_area=300):
        self.first = None
        self.minimum_contour_area = minimum_contour_area

    def __call__(self, index, image, gray):
        if (index == 0):
            self.first = gray
            return image
        frame_delta = cv2.absdiff(self.first, gray)
        thresholded_image = vision.threshold_above(frame_delta, 25)
        thresholded_image = cv2.dilate(thresholded_image, None, iterations=2)
        contours = vision.find_external_contours(thresholded_image)
        contours = [contour for contour in contours if contour.area()
                    > self.minimum_contour_area]
        contours = vision.Contours(contours)
        contours.draw_simple_bounding_boxes(image, color=vision.GREEN)
        return image


class Application:
    def __init__(self, loop):
        self.loop = loop
        # Web cam
        self.webcam = io.WebcamSequence()
        self.detector = MotionDetector()

    def build(self):
        # Frame rate
        fps = 10
        # clock
        clock_source = rx.interval(1/fps)
        webcam = self.webcam
        resize_image = lambda image: vision.resize_by_max_width(image, max_width=500)
        blur = lambda image: filters.gaussian_blur(image, kernel_size=21)
        pipeline = clock_source.pipe(
            ops.map(lambda index: Context(index=index, image=next(webcam).frame)),
            ops.map(step(resize_image, "image", "image")),
            ops.map(step(vision.to_gray, "image", "gray")),
            ops.map(step(blur, "gray", "gray")),
            ops.map(step(self.detector, ["index", "image", "gray"])),
        )
        self.pipeline = pipeline

    def run(self):
        loop = self.loop
        print("Subscribing to the pipeline.")
        subscription = self.pipeline.subscribe(self)
        print("Running event loop forever")
        loop.run_forever()
        print("Application exited from event loop.")
        self.cleanup()

    def on_next(self, context):
        loop = self.loop
        future = asyncio.run_coroutine_threadsafe(self.action(context), loop)
        key = future.result()
        if key == ord('q') or key == 27:
            print("User has requested to stop the application.")
            print("Stopping event loop.")
            loop.stop()

    def on_error(self, e):
        traceback.print_exc()
        self.cleanup()

    def on_completed(self):
        pass

    def cleanup(self):
        print("Stopping web camera.")
        self.webcam.stop()
        print("Destroying all windows.")
        cv2.destroyAllWindows()


    async def action(self, context):
        cv2.imshow("Motion", context.image)
        return cv2.waitKey(1) & 0xff

@click.command()
def main():
    # Event loop
    loop = asyncio.get_event_loop()
    print("Creating application.")
    app = Application(loop)
    print("Building the application pipeline.")
    app.build()
    print("Starting the application pipeline.")
    app.run()
    print("Closing the event loop.")
    loop.close()
    return

if __name__ == '__main__':
    main()
