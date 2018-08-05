'''
Methods for non-maximum suppression of rectangles
'''
import numpy as np

def bb_areas(bounding_boxes):
    '''Returns the areas of bounding boxes
    
    Args:
        bounding_boxes (array): An Nx4 array containing a list of bounding boxes.
            Each box is described by its top left and bottom right coordinates.

    Returns:
        An array of floats containing areas of bounding boxes.    
    '''
    # The top left coordinates
    x_0 = bounding_boxes[:, 0]
    y_0 = bounding_boxes[:, 1]
    # bottom right coordinates
    x_1 = bounding_boxes[:, 2]
    y_1 = bounding_boxes[:, 3]
    # areas of each of the bounding boxes
    areas = (x_1 - x_0 + 1) * (y_1 - y_0 + 1)
    return areas.astype('double')

def non_maximum_suppression(bounding_boxes, scores=None, overlap_threshold=0.3):
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
    '''
    if bounding_boxes.size == 0:
        # There is nothing to do
        return []
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
