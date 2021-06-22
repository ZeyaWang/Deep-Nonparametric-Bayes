import functools
import os, sys
import time
import cv2
import numpy as np
import pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn import mixture
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import moment
from tensorflow.linalg import logdet, trace, inv

import libs.config as cfg
import libs.nets.nets_factory as network 
from libs.load_data import *
from libs.dp.merge import Gibbs_DPM_Gaussian_summary_input
from libs.dp.VI_PYMMG_functions import R_VI_PYMMG_CoC

FLAGS = tf.app.flags.FLAGS

def gmm_loss(ys, mus, gammas):
    """clustering loss L0.
    Args:
        y: nxd tensor: NxD
        mu: nxd tensor; mus multiplied by assign index: NxD
        gamma: dxdxn; precison matrix: NxDxD
    """
    ll = tf.zeros([], dtype=tf.float32)
    def condition(i, ys, mus, gammas, ll):
        r = tf.less(i, tf.shape(ys))
        return r[0]

    def loop(i, ys, mus, gammas, ll):
        y = tf.expand_dims(ys[i], 0) #1xD
        mu = tf.expand_dims(mus[i], 0) #1xD
        gamma = gammas[i] #DxD
        ll = ll  + tf.squeeze(tf.matmul(tf.matmul((y - mu), gamma), 
            tf.transpose(y - mu)))
        return [i+1, ys, mus, gammas, ll]

    i = 0
    [i, ys, mus, gammas, ll] = tf.while_loop(condition, loop,
        [i, ys, mus, gammas, ll])
    return ll/tf.cast(tf.shape(ys)[0], tf.float32)

def standardize(x):
    """standardize a tensor.
    Args:
        x is a nxp tensor
    """
    meanv, varv = tf.nn.moments(x, 0) # p
    stdv = tf.sqrt(varv)
    return (x - meanv)/stdv

def np_standardize(x):
    """standardize a numpy array.
    Args:
        x is a nxp array
    """
    stdv = (moment(x, moment=2,axis=0))**0.5
    meanv = np.mean(x,axis=0)
    return (x - meanv)/stdv, meanv, stdv

def restore(sess, opt=0):
    """restore session with different options
    Args:
        opt = 1: restore from checkpoint
        opt = 0: restore from pretrained initializatoin (remove fc layers)
    """    

    checkpoint_path = FLAGS.checkpoint_path
    vars_to_restore = tf.trainable_variables()
    vars_to_restore1 = vars_to_restore[:]
    if FLAGS.normalize == 1 and opt == 0:
        for var in vars_to_restore1:
            if 'batchnorm' in var.name:
                vars_to_restore.remove(var)
    for var in vars_to_restore1:
        if 'ip' in var.name or 'fc4' in var.name:
            vars_to_restore.remove(var)
    restorer = tf.train.Saver(vars_to_restore)
    restorer.restore(sess, checkpoint_path)

def train():
    ## set the parameters for different datasets
    if FLAGS.dataset == 'mnist_test': 
        img_height = img_width = 28
        learning_rate = 0.001
        Detcoef = 50
        apply_network = 'lenet'
    elif FLAGS.dataset == 'usps': 
        img_height = img_width = 16     
        learning_rate = 0.0001  
        Detcoef = 50
        apply_network = 'lenet0'
    elif FLAGS.dataset == 'frgc': 
        img_height = img_width = 32
        learning_rate = 0.1 
        Detcoef = 20   
        apply_network = 'lenet'
    elif FLAGS.dataset == 'ytf': 
        img_height = img_width = 55
        learning_rate = 0.1
        Detcoef = 20
        apply_network = 'lenet'
    elif FLAGS.dataset == 'umist': 
        img_height = 112
        img_width = 92
        learning_rate = 0.0001
        Detcoef = 20
        apply_network = 'dlenet'
    else:
        img_height = FLAGS.img_height
        img_width = FLAGS.img_width
        learning_rate = FLAGS.learning_rate
        Detcoef = FLAGS.Detcoef
        apply_network = FLAGS.network

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        # tensor for input images
        if FLAGS.is_resize:
            imageip = tf.placeholder(tf.float32, [None, FLAGS.resize_height, FLAGS.resize_width, 3])
        else:    
            imageip = tf.placeholder(tf.float32, [None, img_height, img_width, 3])

        # get the embedding data from the network
        _, end_points =network.get_network(apply_network, imageip, FLAGS.max_k,
            weight_decay=FLAGS.weight_decay, is_training=True, reuse = False, spatial_squeeze=False)
        # fc3 is the name of our embedding layer
        end_net = end_points['fc3']

        # normalize the embedding data
        if FLAGS.normalize==0: # standardize
            end_data = standardize(end_net)
        elif FLAGS.normalize==1: # batch normalize
            end_data = slim.batch_norm(end_net, activation_fn=None, scope='batchnorm',is_training=True)
        
        # calculate LD the sample covaraince variance matrix of embedding data
        diff_data = end_data - tf.expand_dims(tf.reduce_mean(end_data, 0),0)
        cov_data = 1. / (tf.cast(tf.shape(end_data)[0], tf.float32) - 1.)*tf.matmul(tf.transpose(diff_data), diff_data)
        det_loss =  -  logdet(cov_data)

        # get the numpy data for both purpose of clustering and evaluation
        _, val_end_points =network.get_network(apply_network, imageip, FLAGS.max_k,
            weight_decay=FLAGS.weight_decay, is_training=False, reuse = True, spatial_squeeze=False)
        val_end_data = val_end_points['fc3']

        if FLAGS.normalize==1:
            val_end_data = slim.batch_norm(val_end_data, activation_fn=None, scope='batchnorm',is_training=False, reuse=True)

        # clustering loss
        cls_mus = tf.placeholder(tf.float32, [None, FLAGS.embed_dims])
        cls_Gammas = tf.placeholder(tf.float32, [None, FLAGS.embed_dims, FLAGS.embed_dims])
        cluster_loss = gmm_loss(end_data, cls_mus, cls_Gammas)

        # l2 regularization
        penalty = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # total loss
        total_loss = cluster_loss + Detcoef*det_loss
        if penalty:
            l2_penalty = tf.add_n(penalty)
            total_loss += l2_penalty

        global_step = slim.create_global_step()

        ## load the data
        df_path = '{}/{}.h5'.format(FLAGS.dataset_dir, FLAGS.dataset)
        f = h5py.File(df_path, 'r')
        ## Get the data
        data = list(f['data'])
        label = list(f['labels'])
        train_datum = load_train_data(data,label)
        train_datum.center_data()
        train_datum.shuffle(100)
        val_data, val_truth  = np.copy(train_datum.data), np.copy(train_datum.label)

        ## set up mini-batch steps and optimizer
        batch_num = train_datum.data.shape[0]//FLAGS.batch_size


        learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, batch_num, 0.0001*batch_num, True)
        var_list = tf.trainable_variables() 

        opt = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = FLAGS.momentum)
        train_opt = slim.learning.create_train_op(
            total_loss, opt,
            global_step=global_step,
            variables_to_train=var_list,
            summarize_gradients=False)

        ## load session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
                )
        sess.run(init_op)

        ## log setting and results
        timestampLaunch = time.strftime("%d%m%Y") + '-' + time.strftime("%H%M%S")
        # record config
        if not os.path.exists(FLAGS.out_dir):
            os.makedirs(FLAGS.out_dir)
        if not os.path.exists(os.path.join(FLAGS.out_dir, FLAGS.dataset)):
            os.makedirs(os.path.join(FLAGS.out_dir, FLAGS.dataset))
        outdir = os.path.join(FLAGS.out_dir, FLAGS.dataset, timestampLaunch)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if FLAGS.dataset == 'umist':
            max_periods = 2000
        else:
            max_periods = FLAGS.max_periods
        # load saver and restore session
        saver = tf.train.Saver(max_to_keep=3)
        if FLAGS.restore_previous_if_exists:
            restore(sess, 1)
        else:
            if FLAGS.if_initialize_from_pretrain:
                restore(sess, 0)

        period_cluster_l, period_det_l, period_tot_l, conv_cluster_l = [], [], [], [sys.float_info.max]

        """ start the training """ 
        print('start training the dataset of {}'.format(FLAGS.dataset))
        for period in range(max_periods): 
            real_period = period + FLAGS.checkpoint_periods

            '''Forward steps'''
            ## get the numpy array of embedding data for clustering
            val_embed = []
            if FLAGS.dataset == 'mnist_test': #10000
                for s in range(10):
                    start = s*1000
                    end = (s+1)*1000
                    val_embed_x = sess.run(val_end_data, feed_dict={imageip:val_data[start:end]})
                    val_embed.append(val_embed_x)  
            elif FLAGS.dataset == 'usps': # 11000
                for s in range(11):
                    start = s*1000
                    end = (s+1)*1000
                    val_embed_x = sess.run(val_end_data, feed_dict={imageip:val_data[start:end]})
                    val_embed.append(val_embed_x)
            elif FLAGS.dataset == 'frgc': # 2462
                for s in range(25):
                    start = s*100
                    end = (s+1)*100
                    if s == 24:
                        end = end - 38
                    val_embed_x = sess.run(val_end_data, feed_dict={imageip:val_data[start:end]})
                    val_embed.append(val_embed_x)
            elif FLAGS.dataset == 'ytf': ##55x55; 10000
                for s in range(10):
                    start = s*1000
                    end = (s+1)*1000
                    val_embed_x = sess.run(val_end_data, feed_dict={imageip:val_data[start:end]})
                    val_embed.append(val_embed_x)
            elif FLAGS.dataset == 'umist': # < 2000
                val_embed = sess.run(val_end_data, feed_dict={imageip:val_data})
            if FLAGS.dataset != 'umist':
                val_embed = np.concatenate(val_embed,axis=0)

            if FLAGS.normalize==0:
                val_embed, val_mean, val_std = np_standardize(val_embed)
            ### use dpm to cluster the embedding data
            dpgmm = mixture.BayesianGaussianMixture(n_components=FLAGS.max_k, 
                                                   weight_concentration_prior=FLAGS.alpha/FLAGS.max_k,
                                                  weight_concentration_prior_type='dirichlet_process',
                                                  covariance_prior=FLAGS.embed_dims*np.identity(FLAGS.embed_dims),
                                                  covariance_type='full').fit(val_embed)
            val_labels = dpgmm.predict(val_embed)

            if FLAGS.onsign:
                ### SIGN algorithm to merge clusters
                ulabels = np.unique(val_labels).tolist()
                uln_l = []
                ulxtx_l = []
                ulxx_l = []
                for ul in ulabels:
                    ulx = val_embed[val_labels==ul,:] #Nk x p
                    uln = np.sum(val_labels==ul) #Nk
                    ulxtx = np.matmul(ulx.T, ulx) #p x p
                    ulxx = np.sum(ulx, axis=0) # p
                    uln_l.append(uln)
                    ulxtx_l.append(ulxtx)
                    ulxx_l.append(ulxx) 
                uxx = np.stack(ulxx_l, axis=0) #kxp
                un = np.array(uln_l) # k
                uxtx = np.stack(ulxtx_l, axis=0).T # p x p x k

                if FLAGS.embed_dims < 50:
                    Rest = Gibbs_DPM_Gaussian_summary_input(uxtx, uxx, un) # mcmc
                else:
                    Rest = R_VI_PYMMG_CoC(uxtx, uxx, un) # variational inference
                member, dp_Gammas, dp_mus = Rest['member_est'], Rest['Prec'], Rest['mu']
                
                val_labels_new = np.copy(val_labels)
                for u, ul in enumerate(ulabels):
                    val_labels_new[val_labels==ul] = int(member[u]) # order the cluster value with index
                val_labels = np.copy(val_labels_new)   

                # evaluate and save the results                     
                val_count = np.bincount(val_labels)
                val_count2 = np.nonzero(val_count)
                est_cls = {}
                for v in val_count2[0].tolist():
                    est_cls[v] = []
                for vv, vl in enumerate(val_labels.tolist()):
                    est_cls[vl].append(val_truth[vv])
                
                ## sort the labels to be used for backward
                train_labels_new = np.copy(val_labels)
                member1 = np.array([int(m) for m in member])
                member2 = np.unique(member1)
                member2.sort()
                train_labels_new1 = np.copy(train_labels_new)

                for mbi, mb in enumerate(member2.tolist()):
                    train_labels_new1[train_labels_new==mb] = mbi
                train_labels_onehot = np.eye(member2.shape[0])[train_labels_new1]
            else:
                dp_mus = dpgmm.means_
                dp_Gammas = dpgmm.precisions_.T
                train_labels_onehot = np.eye(FLAGS.max_k)[val_labels]

            nmi = normalized_mutual_info_score(val_labels, val_truth)  
            if period > 0:
                print("NMI for period{} is {}".format(period,nmi))

            if period >= 100:
                ## check if the results need to be saved using det_loss and cluster_loss
                dperiod_det_loss = np.abs((period_det_l[-1] - period_det_l[-2])/period_det_l[-2])
                if dperiod_det_loss <= FLAGS.epsilon:
                    conv_cluster_l.append(period_cluster_loss)
                    if conv_cluster_l[-1] < min(conv_cluster_l[:-1]):
                        best_nmi, best_period = nmi, real_period
                        saver.save(sess, os.path.join(outdir, 'ckpt'), real_period)
                        # save truth and labels
                        np.savez(os.path.join(outdir,'labels_{}.npy'.format(real_period)),
                            val_labels=val_labels, val_truth=val_truth,
                            val_mean=val_mean, val_std=val_std)
                        # save dpm model
                        with open(os.path.join(outdir, 'model_{}.pkl'.format(real_period)), 'wb') as pf:
                            pickle.dump(dpgmm, pf)

            if period < max_periods - 1:
                ''' Backward steps'''
                # require: train_labels_onehot:NxK; dp_mus: KxD; dp_Gammas: DxDxK
                train_datum.reset() # reset data from the original order to match predicted label
                period_cluster_loss, period_det_loss = 0., 0.
                for step in range(batch_num):
                    real_step = step + real_period*batch_num                 
                    train_x, train_y = train_datum.nextBatch(FLAGS.batch_size)
                    start, end = step*FLAGS.batch_size, (step+1)*FLAGS.batch_size
                    step_labels_onehot = train_labels_onehot[start:end]
                    cls_mu = np.matmul(step_labels_onehot, dp_mus) # NxK x KxD=> NxD
                    cls_Gamma = np.matmul(dp_Gammas, step_labels_onehot.T).T # DxDxK KxN => DxDxN => NxDxD
                    _, dlossv, dtlossv= sess.run([train_opt, cluster_loss, det_loss], 
                        feed_dict={imageip:train_x, cls_mus:cls_mu, cls_Gammas: cls_Gamma})

                    # save loss
                    period_cluster_loss += dlossv/batch_num
                    period_det_loss += dtlossv/batch_num
                    #print('DP loss for back step {} is {}; det loss is{}, total loss is{}'.format(real_step, 
                    #    dlossv, dtlossv, dlossv + Detcoef*dtlossv))
                ## shuffle train data for next batch
                train_datum.shuffle(period)
                val_data, val_truth  = np.copy(train_datum.data), np.copy(train_datum.label)
                ## record the period loss
                period_tot_loss = period_cluster_loss + Detcoef*period_det_loss
                period_det_l.append(period_det_loss)
                period_cluster_l.append(period_cluster_loss)
                period_tot_l.append(period_tot_loss)



if __name__ == '__main__':
    train()
