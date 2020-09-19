import collections 
import tensorflow as tf
from numpy.linalg import det
import numpy as np
from scipy.stats import multivariate_normal
import cv2

import libs.config as cfg
slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

def get_var_list_to_restore():
  """Choose which vars to restore, ignore vars by setting --checkpoint_exclude_scopes """
  variables_to_restore = []
  if FLAGS.checkpoint_exclude_scopes is not None:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes]
    # build restore list
    for var in tf.model_variables():
      for exclusion in exclusions:
        if var.name.startswith(exclusion):
          break
      else:
        variables_to_restore.append(var)
  else:
    variables_to_restore = tf.model_variables()
  #print(variables_to_restore)
  variables_to_restore_final = []
  if FLAGS.checkpoint_include_scopes is not None:
      includes = [
              scope.strip()
              for scope in FLAGS.checkpoint_include_scopes
              ]
      #print(includes)
      for var in variables_to_restore:
          for include in includes:
              if var.name.startswith(include):
                  variables_to_restore_final.append(var)
                  break
  else:
      variables_to_restore_final = variables_to_restore
  #print(variables_to_restore_final)
  return variables_to_restore_final
