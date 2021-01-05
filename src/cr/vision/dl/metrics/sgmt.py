"""
Metrics related to image segmentation

Derived from keras-unet GITHUB project

"""

import tensorflow as tf

def iou(ground, prediction, smooth=1.0):
    # flatten the tensors
    ground = tf.reshape(ground, [-1])
    prediction = tf.reshape(prediction, [-1])
    # Compute the intersection
    intersection = tf.reduce_sum(ground * prediction)
    # Compute the union
    union  = tf.reduce_sum(ground) + tf.reduce_sum(prediction) - intersection
    # Compute the ratio
    iou = (intersection + smooth) / (union + smooth)
    return iou


def threshold_binarize(x, threshold=0.5):
    # identify indices where x is above threshold
    ge = tf.greater_equal(x, tf.constant(threshold))
    # create an image of 1s and 0s based on threshold
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y


def iou_thresholded(ground, prediction, threshold=0.5, smooth=1.0):
    # threshold the predictions
    prediction = threshold_binarize(prediction, threshold)
    # flatten the tensors
    ground = tf.reshape(ground, [-1])
    prediction = tf.reshape(prediction, [-1])
    # Compute the intersection
    intersection = tf.reduce_sum(ground * prediction)
    # Compute the union
    union  = tf.reduce_sum(ground) + tf.reduce_sum(prediction) - intersection
    # Compute the ratio
    iou = (intersection + smooth) / (union + smooth)
    return iou


def dice_coeff(ground, prediction, smooth=1.0):
    # flatten the tensors
    ground = tf.reshape(ground, [-1])
    prediction = tf.reshape(prediction, [-1])
    # Compute the intersection
    intersection = tf.reduce_sum(ground * prediction)
    # denominator
    denom  = tf.reduce_sum(ground) + tf.reduce_sum(prediction)
    # Compute the ratio
    dice = (2 * intersection + smooth) / (denom + smooth)
    return dice
