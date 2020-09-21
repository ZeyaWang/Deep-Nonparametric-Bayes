import cv2
import numpy as np
import random
import tensorflow as tf
import libs.config as cfg
import os
import pickle
import h5py

FLAGS = tf.app.flags.FLAGS
if FLAGS.dataset in ['frgc']:
    gray_scale = False
else:
    gray_scale = True

def img_rotate(img, degree):
    num_rows,num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols//2, num_rows//2), degree, 1)
    img = cv2.warpAffine(img,rotation_matrix,(num_cols,num_rows))
    return img

def preprocess_img(img, resize=FLAGS.is_resize, 
       mirror=FLAGS.is_mirror, rotate=FLAGS.is_rotate):
    imgs = []
    img = cv2.imread(img)
    if resize:
        img = cv2.resize(img, (FLAGS.resize_width, FLAGS.resize_height), 
            interpolation = cv2.INTER_CUBIC)
    if rotate:
        xx = 4
    else:
        xx = 1
    for r in range(xx):
        degree = (-90) * r #rotation
        tmp = img_rotate(img, degree)
        imgs.append(tmp)
        if mirror:
            tmp1 = np.flip(tmp, 1) 
            imgs.append(tmp1)
    imgs = np.stack(imgs, axis=0)
    return imgs

def extract_tb_label(filename):
    label = os.path.basename(os.path.dirname(filename))
    return int(label)

class load_train_data:
    def __init__(self, data, label):
        self.data = np.stack([d.T if d.shape[0] == 3 else np.concatenate([d,d,d],axis=0).T for d in data], axis=0)
        self.label = np.array(label,dtype=np.int32)
        self.next_index = 0
        self.data_size = self.data.shape[0]

    def center_data(self):
        mean_val = np.mean(self.data, axis=(0,1,2))
        self.data = self.data - mean_val

    def shuffle(self, seeds):
        data_index = list(range(self.data_size))
        random.seed(seeds)
        random.shuffle(data_index)
        self.data = self.data[data_index, :, :, :]
        self.label = np.array([self.label[data_index[i]] for i in range(len(data_index))])    

    def nextBatch(self, batch_size): # extract batch for training 
        if (self.next_index + batch_size) >= (self.data_size):
            datum = self.data[self.next_index:self.data_size, :, :, :]  
            labels = self.label[self.next_index:self.data_size]
            self.next_index = 0
        else:
            datum = self.data[self.next_index:(self.next_index + batch_size), :, :, :]    
            labels = self.label[self.next_index:(self.next_index + batch_size)]
            self.next_index += batch_size              
        return datum, labels

    def reset(self): # reset next index to zero
        self.next_index = 0
         
if __name__ == "__main__":        
    ## Loading image path
    df_path = '../dataset/{}.h5'.format(FLAGS.dataset)
    f = h5py.File(df_path, 'r')
    ## Get the data
    data = list(f['data'])
    label = list(f['labels'])
    train_datum = load_train_data(data,label)
    train_datum.center_data()
    train_datum.shuffle(100)
    with open(df_path[:-3]+'.p', 'wb') as pf:
        pickle.dump((train_datum), pf)
        