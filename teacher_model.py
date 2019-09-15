import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
import tensorflow as tf
import numpy as np
import tf_util


def placeholder_inputs(batch_size, num_pixels, num_classes):

    images_pl = tf.placeholder(tf.float32, shape=(batch_size, num_pixels))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_classes))

    return images_pl, labels_pl


def get_model(images, is_training, num_classes, bn_decay=None):
    fc1 = tf_util.fully_connected(images, 1200, bn_decay=bn_decay, is_training=is_training, scope='fc1')
    fc2 = tf_util.fully_connected(fc1, 1200, bn_decay=bn_decay, is_training=is_training, scope='fc2')
    fc2 = tf_util.dropout(fc2, is_training=is_training, scope='dp')
    logits = tf_util.fully_connected(fc2, num_classes, bn_decay=bn_decay, is_training=is_training, scope='logits')
    return logits


def get_loss(image_pred, labels):
    return tf.nn.softmax_cross_entropy_with_logits(image_pred, labels)




