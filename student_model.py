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
    soft_labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_classes))

    return images_pl, labels_pl, soft_labels_pl


def get_model(images, is_training, num_classes, T, bn_decay=None):
    fc1 = tf_util.fully_connected(images, 800, bn_decay=bn_decay, is_training=is_training, scope='fc1')
    fc1 = tf_util.dropout(fc1, is_training=is_training, scope='dp')
    logits = tf_util.fully_connected(fc1, num_classes, bn_decay=bn_decay, is_training=is_training, scope='logits')
    T = tf.cond(is_training == True, lambda: tf.constant(T), lambda: tf.constant(1.))
    logits = logits / T
    return logits


def get_loss(image_pred, hard_labels, soft_labels, T):
    hard_loss = tf.nn.softmax_cross_entropy_with_logits(logits=image_pred * T, labels=hard_labels)
    hard_loss = tf.reduce_mean(hard_loss)

    soft_labels = tf.nn.softmax(soft_labels)
    image_pred = tf.nn.softmax(image_pred)
    sof_image_log = tf.log(image_pred)
    soft_loss = tf.multiply(soft_labels, sof_image_log)
    soft_loss = tf.reduce_sum(soft_loss)
    soft_loss = -1 * soft_loss

    loss = (hard_loss + T * T * soft_loss) / 2

    return loss



