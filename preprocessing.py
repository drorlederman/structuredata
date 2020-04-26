#import keras
#from keras import backend as K
#from keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
#from utils import scale
import tensorflow as tf
import torch
import numpy as np
from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit

def train_val_test_split(data_vect, labels, train_size=0.7, val_size=0.15, test_size=0.15):
    x_train_vect, x_test_vect, y_train, y_test = train_test_split(data_vect, labels, test_size=(val_size + test_size))
    x_val_vect, x_test_vect, y_val, y_test = train_test_split(x_test_vect, y_test,
                                                              test_size=test_size / (1 - train_size))
    return x_train_vect, x_val_vect, x_test_vect, y_train, y_val, y_test

def train_val_test_split_dataset(dataset, train_size=0.7, valid_size=0.15, test_size=0.15):

    train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

    x_train_vect, x_test_vect, y_train, y_test = train_test_split(dataset, labels, test_size=(valid_size + test_size))
    x_val_vect, x_test_vect, y_val, y_test = train_test_split(x_test_vect, y_test,
                                                              test_size=test_size / (1 - train_size))
    return x_train_vect, x_val_vect, x_test_vect, y_train, y_val, y_test

def my_train_test_split(data_size, test_size=0.30):
    sss = ShuffleSplit(n_splits=1, test_size=test_size)
    X = np.reshape(np.random.rand(data_size * 2), (data_size, 2))
    y = np.random.randint(2, size=data_size)
    sss.get_n_splits(X, y)
    train_index, test_index = next(sss.split(X, y))
    return train_index, test_index

def my_train_val_test_split(data_size, train_size=0.7, valid_size=0.15, test_size=0.15):
    train_index, test_index = my_train_test_split(data_size, test_size=(valid_size + test_size))
    val_index, test_index = my_train_test_split(test_size, test_size=test_size / (1 - train_size))
    return train_index, val_index, test_index

def pre_process(data, img_rows, img_cols, labels, num_classes, one_hot):
    a = False
    if a:
        data = data.reshape(data.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        data = data.reshape(data.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    data = data.astype('float32')
    # convert class vectors to binary class matrices
    if one_hot:
        labels = tf.keras.utils.to_categorical(labels, num_classes)

    data_mn, data_std = np.mean(data), np.std(data)

    return data, labels, input_shape, data_mn, data_std