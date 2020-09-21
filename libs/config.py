import tensorflow as tf

##########################
#       Clustering       #
##########################
tf.app.flags.DEFINE_float(
    'alpha', 1.,
    'wight concentration prior in dpm')
tf.app.flags.DEFINE_bool(
    'onsign', True,
    'if SIGN is activated')
tf.app.flags.DEFINE_integer(
    'max_k', 100,
    'maximum number of clusters')

##########################
#     Save directory     #
##########################
tf.app.flags.DEFINE_string(
    'out_dir', './output',
    'output directory')

##########################
#         network        #
##########################
tf.app.flags.DEFINE_float(
    'dropout_keep_prob', 0.5, 
    'dropout probability for nodes.')
tf.app.flags.DEFINE_string(
    'network', 'lenet',
    'backbone network')
tf.app.flags.DEFINE_integer(
    'embed_dims', 20,
    'embedded dimensions')
tf.app.flags.DEFINE_string(
    'dlenet_filter', '32,32,64,64',
    'cnn filter dimension')
tf.app.flags.DEFINE_integer(
    'dlenet_filter_size', 5,
    'cnn filter dimension')

##########################
#    Optimization Flags  #
##########################
tf.app.flags.DEFINE_float(
    'Detcoef', 20.,
    'balance parameter for repulsion')
tf.app.flags.DEFINE_integer(
    'max_periods', 500,
    'max periods')
tf.app.flags.DEFINE_integer(
    'normalize', 0,
    'normalization methods')
tf.app.flags.DEFINE_float(
    'weight_decay', 0.0001, 
    'weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'momentum for the MomentumOptimizer')
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
    'initial learning rate.')
tf.app.flags.DEFINE_float(
    'epsilon', 0.01,
    'relative tolerance')

#######################
#     Dataset Flags   #
#######################
tf.app.flags.DEFINE_string(
    'dataset', 'usps', 
    'dataset name')
tf.app.flags.DEFINE_integer(
    'resize_width', 32,
    'width after resizing' )
tf.app.flags.DEFINE_integer(
    'resize_height', 32,
    'height after resizing' )
tf.app.flags.DEFINE_integer(
    'img_width', 16,
    'width' )
tf.app.flags.DEFINE_integer(
    'img_height', 16,
    'height' )
tf.app.flags.DEFINE_string(
    'dataset_dir', './dataset',
    'the directory where the dataset files are stored.')

#####################
# Training Flags #
#####################
tf.app.flags.DEFINE_integer(
    'batch_size', 256,
    'the number of samples in each batch.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'the path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_integer(
    'checkpoint_periods', 0,
    'the restored checkpoint period to start with')
tf.app.flags.DEFINE_boolean(
    'restore_previous_if_exists', False,
    'if restoring a checkpoint')
tf.app.flags.DEFINE_boolean(
    'if_initialize_from_pretrain', False,
    'if initialized from pretrained results.')

###################
# Data-Preprocess #
###################
tf.app.flags.DEFINE_boolean(
    'is_rotate', False,
    'rotate image or not')
tf.app.flags.DEFINE_boolean(
    'is_resize', False,
    'resize image or not')
tf.app.flags.DEFINE_boolean(
    'is_mirror', False,
    'mirror image or not')

FLAGS = tf.app.flags.FLAGS