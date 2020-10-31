'''
Helper functions and classes for contour finding, drawing, analysis
'''
import numpy as np
import cv2
from . import colors as iv_colors


class Contour:
    '''
    Various parameters related to a contour
    '''

    def __init__(self, contour):
        self._contour = contour
        self._moments = cv2.moments(contour)

    def moments(self):
        '''Returns all the moments for a contour'''
        return self._moments

    def centroid(self):
        '''Returns the centroid of a contour'''
        moments = self._moments
        c_x = int(moments['m10']/moments['m00'])
        c_y = int(moments['m01']/moments['m00'])
        return (c_x, c_y)

    def area(self):
        '''Returns the area of the contour'''
        return cv2.contourArea(self._contour)

    def perimeter(self):
        '''Returns the perimeter of the contour'''
        return cv2.arcLength(self._contour, True)

    def approximate_polygon(self, perimeter_gap_factor=0.1):
        '''Returns an approximate shape using the Douglas-Peucker algorithm'''
        epsilon = perimeter_gap_factor * self.perimeter()
        return cv2.approxPolyDP(self._contour, epsilon, True)

    def is_convex(self):
        '''Returns if the contour is convex'''
        return cv2.isContourConvex(self._contour)

    def convex_hull_points(self, clockwise=False):
        '''Returns the points which form the convex hull of a contour
        using the Sklansky's algorithm'''
        return cv2.convexHull(self._contour, clockwise=clockwise)

    def convex_hull_indices(self, clockwise=False):
        '''Returns the indices of the points which form the convex hull of a contour
        using the Sklansky's algorithm'''
        return cv2.convexHull(self._contour, clockwise=clockwise, returnPoints=False)

    def simple_bounding_box(self):
        '''Returns the bounding box of the contour which is straight rectangle
        without considering the orientation of object'''
        return cv2.boundingRect(self._contour)

    def best_fit_bounding_box(self):
        '''Returns the rotated bounding box considering the orientation of object'''
        min_rect = cv2.minAreaRect(self._contour)
        box = cv2.boxPoints(min_rect)
        box = np.intp(box)
        return box

    def best_fit_circle(self):
        '''Returns the minimum enclosing circle around the contour'''
        (c_x, c_y), radius = cv2.minEnclosingCircle(self._contour)
        center = (c_x, c_y)
        return center, radius

    def best_fit_ellipse(self):
        '''Returns the best fitting ellipse around the contour'''
        ellipse = cv2.fitEllipse(self._contour)
        return ellipse

    def best_fit_line(self):
        '''Returns the best fitting line around the contour'''
        (v_x, v_y, x_0, y_0) = cv2.fitLine(
            self._contour, cv2.DIST_L2, 0, 0.01, 0.01)
        unit_vector = (v_x, v_y)
        point = (x_0, y_0)
        return unit_vector, point


def find_external_contours(image, method=cv2.CHAIN_APPROX_SIMPLE):
    '''Finds the external outer contours in a given (grayscale) image'''
    result = cv2.findContours(image, mode=cv2.RETR_EXTERNAL,
                              method=method)
    contours = result[1]
    return [Contour(contour) for contour in contours]


class Contours:
    '''Helper class to work with a list of contours'''

    def __init__(self, contours):
        self._contours = contours

    def draw_centroids(self, image, color=iv_colors.BLACK, marker_size=10, thickness=2):
        '''Draws the centroids of contours on the given image'''
        centroids = [contour.centroid() for contour in self._contours]
        marker_type = cv2.MARKER_CROSS
        for centroid in centroids:
            cv2.drawMarker(image, centroid, color, marker_type,
                           markerSize=marker_size, thickness=thickness)

    def draw_simple_bounding_boxes(self, image, color=iv_colors.BLACK, thickness=2):
        '''Draws simple bounding boxes around the contours in a given image'''
        bounding_boxes = [contour.simple_bounding_box()
                          for contour in self._contours]
        for bounding_box in bounding_boxes:
            left, top, width, height = bounding_box
            top_left = (left, top)
            bottom_right = (left + width, top + height)
            cv2.rectangle(image, top_left, bottom_right,
                          color, thickness=thickness)

    def draw_best_fit_bounding_boxes(self, image, color=iv_colors.BLACK, thickness=2):
        '''Draws the best fit (rotated) bounding boxes around contours'''
        bounding_boxes = [contour.best_fit_bounding_box()
                          for contour in self._contours]
        cv2.drawContours(image, bounding_boxes, -1, color, thickness)
