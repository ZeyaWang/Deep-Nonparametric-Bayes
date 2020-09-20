from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import tensorflow as tf
import tensorflow.contrib.slim as slim
import libs.config as cfg
from libs.nets.lenet import lenet, dlenet, lenet0

FLAGS = tf.app.flags.FLAGS


def get_network(name, image, num_classes, weight_decay, is_training, reuse, dropout_keep_prob=0.5, spatial_squeeze=True, scope=None):
    if name[:5] == 'lenet':
        with slim.arg_scope(vgg_arg_scope(weight_decay=weight_decay)):
             logits, end_points = lenet(image, num_classes, is_training, dropout_keep_prob, reuse = reuse)
        return logits, end_points
    elif name[:6] == 'lenet0':
        with slim.arg_scope(vgg_arg_scope(weight_decay=weight_decay)):
             logits, end_points = lenet0(image, num_classes, is_training, dropout_keep_prob, reuse = reuse)
        return logits, end_points
    elif name[:6] == 'dlenet':
        with slim.arg_scope(vgg_arg_scope(weight_decay=weight_decay)):
             logits, end_points = dlenet(image, num_classes, is_training, dropout_keep_prob, reuse = reuse)
        return logits, end_points




