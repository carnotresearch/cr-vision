
import os
import csv
import numpy as np
import logging
import logging.handlers
import math
import sys
import random
import numpy as np
import skvideo.io
import cv2
import matplotlib.pyplot as plt

from IPython.display import HTML
from base64 import b64encode

from cr import vision

_ELLIPSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

def _filter_mask(mask_image):
    # Fill any small holes
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, _ELLIPSE_KERNEL)
    # Remove noise
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, _ELLIPSE_KERNEL)
    # Dilate to merge adjacent blobs
    mask_image = cv2.dilate(mask_image, _ELLIPSE_KERNEL, iterations=2)    
    return mask_image



class TrafficCounter(object):
    """Counts the number of vehicles in a traffic video
    """

    WARMUP_FRAME_COUNT = 500
    MIN_CONTOUR_WIDTH = 35
    MIN_CONTOUR_HEIGHT = 35


    def __init__(self, source, warmup_frame_count=None, 
        min_contour_width=None,
        min_contour_height=None):
        self.source = source
        self.cap = skvideo.io.vreader(source)
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, detectShadows=True)
        self.fg_mask = None
        if warmup_frame_count is not None:
            self.WARMUP_FRAME_COUNT = warmup_frame_count
        if min_contour_width is not None:
            self.MIN_CONTOUR_WIDTH = min_contour_width
        if min_contour_height is not None:
            self.MIN_CONTOUR_HEIGHT = min_contour_height
        pass

    def warmup(self):
        i = 0
        bg = self.bg_sub
        for frame in self.cap:
            self.fg_mask = bg.apply(frame, None, 0.001)
            i += 1
            print(".", end ="", flush=True)
            if i >= self.WARMUP_FRAME_COUNT:
                break

    def next_frame(self):
        frame = next(self.cap)
        # keep a copy of frame for later reference
        self.frame  = frame.copy()
        # Perform background subtraction
        fg_mask = self.bg_sub.apply(frame, None, 0.001)
        self.fg_mask = fg_mask
        # Cleanup the subtracted image
        # ignore small values
        fg_mask[fg_mask < 240] = 0
        # filter 
        filtered_mask = _filter_mask(fg_mask)
        # Keep the filtered image
        self.filtered_mask = filtered_mask
        # Find vehicles in image
        self._find_vehicles()
        return filtered_mask


    def _find_vehicles(self):
        mask = self.filtered_mask
        matches = []
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2:]
        for (i, contour) in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            valid = (w > self.MIN_CONTOUR_WIDTH) and (h > self.MIN_CONTOUR_HEIGHT)
            if not valid: 
                continue
            # Add this has an identified vehicle
            matches.append((x, y, w, h))
        self.matches = matches
        return matches


    def draw_vehicles(self):
        matches = self.matches
        frame  = self.frame.copy()
        for match in matches:
            (x1, y1, w, h) = match
            x2 = x1 + w
            y2 = y1 + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), vision.RED, 2)
        return frame






