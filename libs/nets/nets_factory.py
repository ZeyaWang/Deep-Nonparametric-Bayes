from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import tensorflow as tf
import tensorflow.contrib.slim as slim
import libs.config as cfg
from libs.nets.lenet import lenet, dlenet, lenet0
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops

FLAGS = tf.app.flags.FLAGS

def net_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.
  Args:
    weight_decay: The l2 regularization coefficient.
  Returns:
    An arg_scope.
  """
  with arg_scope(
      [layers.conv2d, layers_lib.fully_connected],
      activation_fn=nn_ops.relu,
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      biases_initializer=init_ops.zeros_initializer()):
    with arg_scope([layers.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def get_network(name, image, num_classes, weight_decay, is_training, reuse, dropout_keep_prob=0.5, spatial_squeeze=True, scope=None):
    if name[:5] == 'lenet':
        with slim.arg_scope(net_arg_scope(weight_decay=weight_decay)):
             logits, end_points = lenet(image, num_classes, is_training, dropout_keep_prob, reuse = reuse)
        return logits, end_points
    elif name[:6] == 'lenet0':
        with slim.arg_scope(net_arg_scope(weight_decay=weight_decay)):
             logits, end_points = lenet0(image, num_classes, is_training, dropout_keep_prob, reuse = reuse)
        return logits, end_points
    elif name[:6] == 'dlenet':
        with slim.arg_scope(net_arg_scope(weight_decay=weight_decay)):
             logits, end_points = dlenet(image, num_classes, is_training, dropout_keep_prob, reuse = reuse)
        return logits, end_points




