import cv2
import click
import numpy as np
from cr import vision
from dataclasses import dataclass
import rx
import rx.operators as ops
from cr.vision.crx import step
from pathlib import Path



def resize_to_max_width(image):
    width = image.shape[1]
    width = min(600, width)
    result = vision.resize_by_width(image, target_width=width)
    return result

@dataclass
class Context:
    filepath : Path = None
    image : np.ndarray = None
    gray : np.ndarray = None
    faces : list = None

class Subsciber:
    def __init__(self):
        self.count = 0

    def on_next(self, context):
        self.count += 1
        print("{}: {}, {}".format(self.count, context.filepath, type(context)))
        cv2.imshow("Faces", context.image)
        key = cv2.waitKey(0) & 0xFF

    def on_error(self, e):
        print(e)
    def on_completed(self):
        print("done")

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--count', default=10, help='Number of images to process.')
def main(input_dir, count):
    extensions = [".jpg", ".png"]
    input_dir  = Path(input_dir).resolve()
    # image reader from Path
    reader = lambda path: cv2.imread(str(path))
    # face detector
    detector = vision.faces.CascadeDetector()
    source = rx.from_iterable(input_dir.iterdir()).pipe(
        ops.filter(lambda path: path.is_file()),
        ops.filter(lambda path: path.suffix in extensions),
        ops.take(count),
        ops.map(lambda path: Context(filepath=path)),
        ops.map(step(reader, "filepath", "image")),
        ops.map(step(resize_to_max_width, "image", "image")),
        ops.map(step(vision.to_gray, "image", "gray")),
        ops.map(step(detector, "gray", "faces")),
        ops.map(step(vision.bb.draw_boxes, ["image", "faces"]))
        )
    subscription = source.subscribe(Subsciber())
    return

if __name__ == '__main__':
    main()