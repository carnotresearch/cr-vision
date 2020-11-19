import numpy as np
import cv2
from cr import vision
from . import colors

def draw_boxes(image, boxes, color=colors.RED, box_type="xywh"):
    if box_type == 'xywh':
        for (x, y, w, h) in boxes:
            cv2.rectangle(image, (x, y), (x+w-1, y+h-1), color, 2)
        return
    if box_type == 'ltrb':
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        return
    raise NotImplemented

def areas(boxes, box_type="xywh"):
    '''Returns the areas of bounding boxes
    
    Args:
        boxes (array): An Nx4 array containing a list of bounding boxes.
            Each box is described by its top left and bottom right coordinates.

    Returns:
        An array of floats containing areas of bounding boxes.    
    '''
    # The top left coordinates
    if box_type == 'xywh':
        w = boxes[:, 2]
        h = boxes[:, 3]
        areas = w * h
        return areas.astype('double')
    if box_type == 'ltrb':
        x_0 = boxes[:, 0]
        y_0 = boxes[:, 1]
        # bottom right coordinates
        x_1 = boxes[:, 2]
        y_1 = boxes[:, 3]
        # areas of each of the bounding boxes
        areas = (x_1 - x_0 + 1) * (y_1 - y_0 + 1)
        return areas.astype('double')
    raise NotImplemented


def xywh_to_ltrb(boxes):
    if boxes is None or len(boxes) == 0:
        return boxes
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    result = boxes
    result[:, 2] = x + w - 1
    result[:, 3] = y + h - 1
    return result


def ltrb_to_xywh(boxes):
    if boxes is None or len(boxes) == 0:
        return boxes
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    result = boxes
    result[:, 2] = x1 - x0 + 1
    result[:, 3] = y1 - y0 + 1
    return result


def nms(bounding_boxes, scores=None, overlap_threshold=0.3):
    '''Suppresses smaller boxes in the surroundings.

    Args:
        bounding_boxes (array): An Nx4 array containing a list of bounding boxes.
            Each box is described by its top left and bottom right coordinates.

    Kwargs:
        scores (array): Confidence scores of each bounding box.
            If scores are not provided, then bounding boxes are selected on the basis of their areas.
            If scores are provided then bounding boxes are selected on the basis of their scores.
        overlap_threshold (float): Overlap threshold for suppression. 
            For a selected box b_i, all the boxes b_j that are covered by more than the
            overlap threshold are suppressed.

    Returns:
        Selected bounding boxes which were not suppressed.

    .. todo::

        * Need to consider only those rectangles which are nearby.

    '''
    if bounding_boxes is None or len(bounding_boxes) == 0 or  bounding_boxes.size == 0:
        # There is nothing to do
        return bounding_boxes
    # The top left coordinates
    x_0 = bounding_boxes[:, 0]
    y_0 = bounding_boxes[:, 1]
    # bottom right coordinates
    x_1 = bounding_boxes[:, 2]
    y_1 = bounding_boxes[:, 3]
    # areas of each of the bounding boxes
    areas = (x_1 - x_0 + 1) * (y_1 - y_0 + 1)
    if scores is not None:
        # sort the bounding boxes by confidence scores
        indices = np.argsort(scores)
    else:
        # sort the bounding boxes by areas
        indices = np.argsort(areas)
    # selected bounding boxes list
    selected = []
    while indices.size > 0:
        # pick up the last bounding box
        last = len(indices) - 1
        # index of the last bounding box
        i = indices[last]
        selected.append(i)
        # remaining indices
        remaining = indices[:last]
        # coordinates of overlap between last box and remaining boxes
        xx_0 = np.maximum(x_0[i], x_0[remaining])
        yy_0 = np.maximum(y_0[i], y_0[remaining])
        xx_1 = np.minimum(x_1[i], x_1[remaining])
        yy_1 = np.minimum(y_1[i], y_1[remaining])
        # compute the width and height of overlap
        overlap_widths = np.maximum(0, xx_1 - xx_0 + 1)
        overlap_heights = np.maximum(0, yy_1 - yy_0 + 1)
        # compute the area of overlap
        overlap_areas = overlap_widths * overlap_heights
        # if area is 0, that means there is no overlap
        # compute the overlap ratios for all remaining bounding boxes
        overlap_ratios = overlap_areas.astype('float') / areas[remaining]
        to_be_suppressed = np.where(overlap_ratios > overlap_threshold)
        # Remove the suppressed boxes from the list of indices
        indices = np.delete(remaining, to_be_suppressed)
    return bounding_boxes[selected]
