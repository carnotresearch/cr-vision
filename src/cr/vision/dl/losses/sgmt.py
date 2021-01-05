"""
Losses suitable for image segmentation tasks
"""
import tensorflow as tf


def jaccard_distance(ground, prediction, smooth=100):
    # flatten the tensors
    ground = tf.reshape(ground, [-1])
    prediction = tf.reshape(prediction, [-1])
    # Compute the intersection
    intersection = tf.reduce_sum(ground * prediction)
    # Compute the union
    union  = tf.reduce_sum(ground) + tf.reduce_sum(prediction) - intersection
    # Compute the ratio
    iou = (intersection + smooth) / (union + smooth)
    return (1 - iou) * smooth
