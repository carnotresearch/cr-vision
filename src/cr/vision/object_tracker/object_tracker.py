'''
Base class for object trackers
'''

import cv2

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

OPENCV_CSRT_TRACKER = "csrt"
OPENCV_KCF_TRACKER = "kcf"
OPENCV_BOOSTING_TRACKER = "boosting"
OPENCV_MIL_TRACKER = "mil"
OPENCV_TLD_TRACKER = "tld"
OPENCV_MEDIANFLOW_TRACKER = "medianflow"
OPENCV_MOSSE_TRACKER = "mosse"

OBJECT_TRACKERS = {
    OPENCV_CSRT_TRACKER: cv2.TrackerCSRT_create,
    OPENCV_KCF_TRACKER: cv2.TrackerKCF_create,
    #OPENCV_BOOSTING_TRACKER: cv2.TrackerBoosting_create,
    OPENCV_MIL_TRACKER: cv2.TrackerMIL_create,
    #OPENCV_TLD_TRACKER: cv2.TrackerTLD_create,
    #OPENCV_MEDIANFLOW_TRACKER: cv2.TrackerMedianFlow_create,
    #OPENCV_MOSSE_TRACKER: cv2.TrackerMOSSE_create
}


class ObjectTracker:
    '''Base class for object trackers'''

    tracker = None
    bounding_box = None

    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, bounding_box):
        '''Initializes the tracker'''
        assert self.tracker is not None
        self.bounding_box = bounding_box
        return self.tracker.init(image, bounding_box)

    def update(self, image):
        '''Updates the bounding box for next frame'''
        assert self.tracker is not None
        result, bounding_box = self.tracker.update(image)
        if result:
            self.bounding_box = bounding_box
            return bounding_box
        else:
            return None


def create_object_tracker(tracker_type):
    '''Constructs a tracker object'''
    if tracker_type not in OBJECT_TRACKERS:
        # OpenCV doesn't support this tracker
        return None
    constructor = OBJECT_TRACKERS[tracker_type]
    tracker = constructor()
    return ObjectTracker(tracker)
