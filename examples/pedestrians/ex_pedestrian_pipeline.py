# Command line processing
import click
import traceback
from pathlib import Path
# Vision related imports
import cv2
import numpy as np
from cr import vision
# Rx related imports
from dataclasses import dataclass
import rx
import rx.operators as ops
from cr.vision.crx import step


@dataclass
class Context:
    filepath : Path = None
    image : np.ndarray = None
    all_rectangles : list = None
    nms_rectangles : list = None
    weights : list = None

class Application:
    def __init__(self, input_dir, total):
        self.count = 0
        self.input_dir = Path(input_dir).resolve()
        self.total = total
        # image reader from Path
        self.reader = lambda path: cv2.imread(str(path))
        # initialize the HOG descriptor/person detector
        self.detector = vision.pedestrians.HOGDetector()

    def build(self):
        extensions = [".jpg", ".png"]
        pipeline = rx.from_iterable(self.input_dir.iterdir()).pipe(
            ops.filter(lambda path: path.is_file()),
            ops.filter(lambda path: path.suffix in extensions),
            ops.take(self.total),
            ops.map(lambda path: Context(filepath=path)),
            ops.map(step(self.reader, "filepath", "image")),
            ops.map(step(vision.resize_by_max_width, "image", "image")),
            ops.map(step(self.detector, "image", ["all_rectangles", "weights"])),
            ops.map(step(lambda boxes: vision.bb.nms(boxes, overlap_threshold=0.7), 
                "all_rectangles", "nms_rectangles")),
            ops.map(step(vision.bb.draw_boxes, ["image", "nms_rectangles"]))
            )
        self.pipeline = pipeline

    def run(self):
        subscription = self.pipeline.subscribe(self)
        subscription.dispose()

    def on_next(self, context):
        self.count += 1
        print("{}: {}, {}".format(self.count, context.filepath, type(context)))
        cv2.imshow("Faces", context.image)
        key = cv2.waitKey(0) & 0xFF

    def on_error(self, e):
        traceback.print_exc()

    def on_completed(self):
        print("done")


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--count', default=10, help='Number of images to process.')
def main(input_dir, count):
    app = Application(input_dir, count)
    app.build()
    app.run()
    return



if __name__ == '__main__':
    main()