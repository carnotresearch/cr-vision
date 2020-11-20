import traceback
# Command line processing
import click
from pathlib import Path
# Vision related imports
import cv2
import numpy as np
from cr import vision
# Rx related imports
from dataclasses import dataclass
import rx
import rx.operators as ops
from cr.vision.crx import step, log_tic, log_toc

@dataclass
class Context:
    frame : vision.video.Frame = None
    image : np.ndarray = None
    all_rectangles : list = None
    nms_rectangles : list = None
    weights : list = None
    detection_start : int = 0
    detection_end: int = 0
    nms_start: int = 0
    nms_end: int = 0

class Subsciber:
    def __init__(self):
        self.count = 0

    def on_next(self, context):
        self.count += 1
        detection_time = (context.detection_end - context.detection_start) / (10**6)
        nms_time = (context.nms_end - context.nms_start) / (10**6)
        print("{}, detection: {} ms, nms: {} ms".format(self.count, detection_time, nms_time))
        cv2.imshow("Pedestrians", context.image)
        key = cv2.waitKey(1) & 0xFF

    def on_error(self, e):
        traceback.print_exc()

    def on_completed(self):
        print("done")



@click.command()
@click.argument('input_video_file', type=click.Path(exists=True))
@click.argument('output_video_file', type=click.Path(exists=False), required=False, default=None)
@click.option('--count', default=100, help='Number of frames to process.')
def main(input_video_file, output_video_file, count):
    video_src = vision.VideoFileSequence(input_video_file)
    src_width = video_src.width
    src_height = video_src.height
    print("Source resolution: {}x{}".format(src_width, src_height))
    # initialize the HOG descriptor/person detector
    detector = vision.pedestrians.HOGDetector()
    source = rx.from_iterable(video_src).pipe(
        ops.map(lambda frame: Context(frame=frame, image=frame.frame)),
        ops.map(step(vision.resize_by_max_width, "image", "image")),
        ops.map(log_tic("detection_start")),
        ops.map(step(detector, "image", ["all_rectangles", "weights"])),
        ops.map(log_tic("detection_end")),
        ops.map(log_tic("nms_start")),
        ops.map(step(lambda boxes: vision.bb.nms(boxes, overlap_threshold=0.3), 
            "all_rectangles", "nms_rectangles")),
        ops.map(log_tic("nms_end")),
        ops.map(step(vision.bb.draw_boxes, ["image", "nms_rectangles"]))
        )
    writer = None
    if output_video_file is not None:
        writer = vision.io.VideoWriter(output_video_file,
            fps=video_src.fps,
            frame_size=(src_width, src_height)
            )
        source = source.pipe(
            ops.map(step(lambda image: cv2.resize(image, (src_width, src_height)),
                "image", "image")),
            ops.map(step(writer, "image")),
            )
    if count is not None:
        source = source.pipe(ops.take(count))
    subscription = source.subscribe(Subsciber())
    if writer is not None:
        writer.stop()
    return



if __name__ == '__main__':
    main()
