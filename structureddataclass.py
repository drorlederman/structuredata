from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
import math
from multiprocessing import Pool
import tensorflow as tf
#import tensorflow.keras.preprocessing.image.ImageDataGenerator as ImageDataGenerator
#from tf.keras.preprocessing.image import ImageDataGenerator
from utils import scramble_image, scale

class StructDataTransformer(TransformerMixin):
    """Transforms structured data
    """

    def __init__(self, max_std=None, to_codes=None, ignore=None, randomize_features=False, indices=[],
                 convert_rgb=False):
        self.max_std = max_std
        if not to_codes:
            to_codes = []
        if not ignore:
            ignore = []
        self.ignore_ = ignore
        self.image_dim = 0
        self.padded_vect = []
        self.img = []
        self.imgs = []
        self.labels = []
        self.augmented_imgs = []
        self.augmented_labels = []
        self.combined_training_imgs = []
        self.combined_training_labels = []
        self.means = []
        self.stds = []
        self.imgs_normalized = []
        self.randomize_features = randomize_features
        self.new_order = []
        self.indices = indices
        self.convert_rgb = convert_rgb
        return

    def randomize_features(self, X):
        # function accepts vector
        if self.randomize_features and len(self.new_order) == 0:
            self.new_order = np.random.permutation(X.shape[1])
        return

    def permute(self, X, order):
        X = X[:, order]
        return X

    def combine_data(self, x_train, y_train):
        if (len(self.augmented_imgs) > 0) & (len(self.augmented_labels) > 0):
            self.combined_training_imgs = np.concatenate((x_train, self.augmented_imgs), axis=0)
            self.combined_training_labels = np.concatenate((y_train, self.augmented_labels), axis=0)
        else:
            self.combined_training_imgs = x_train.copy()
            self.combined_training_labels = y_train.copy()
        return

    def augment(self, x_train, y_train, no_of_batches=20, no_of_scrambles=0, batch_size=256):
        if (no_of_batches > 0):
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True)
            datagen.fit(x_train)
            train_generator = datagen.flow(x_train, y_train,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           seed=42)
            i = 0
            self.augmented_imgs = []
            self.augmented_labels = []
            for batch in train_generator:
                i += 1
                x_batch, y_batch = train_generator.next()
                x_batch = np.expand_dims(x_batch, axis=0)
                y_batch = np.expand_dims(y_batch, axis=0)
                if (len(self.augmented_imgs) == 0):
                    for row_batch, row_label in zip(x_batch, y_batch):
                        self.augmented_imgs = row_batch
                        self.augmented_labels = row_label
                else:
                    for row_batch, row_label in zip(x_batch, y_batch):
                        self.augmented_imgs = np.append(self.augmented_imgs, row_batch, axis=0)
                        self.augmented_labels = np.append(self.augmented_labels, row_label, axis=0)
                if i > no_of_batches:
                    break
        if (no_of_scrambles > 0):
            for row, row_label in zip(x_train, y_train):
                for i in range(0, no_of_scrambles):
                    img = scramble_image(row)
                    img = np.expand_dims(img, axis=0)
                    img_label = np.expand_dims(row_label, axis=0)
                    if (len(img_label.shape) == 1):
                        img_label = np.expand_dims(img_label, axis=0)
                    if (len(self.augmented_imgs) == 0):
                        self.augmented_imgs = img
                        self.augmented_labels = img_label
                    else:
                        self.augmented_imgs = np.append(self.augmented_imgs, img, axis=0)
                        # print(self.augmented_labels.shape, img_label.shape)
                        self.augmented_labels = np.append(self.augmented_labels, img_label, axis=0)
        return self

    def calc_padding_order(self, X):
        return self.image_dim ** 2 - X.shape[1]

    def calc_stats(self):
        self.means = self.imgs.mean(axis=(0), dtype='float64')
        self.stds = self.imgs.std(axis=(0), dtype='float64')
        return

    def pad(self, X):
        self.image_dim = math.ceil(math.sqrt(X.shape[1]))
        # self.image_dim = 28      # TBD
        no_of_zeros = self.calc_padding_order(X)
        x_padded = np.pad(X, ((0, 0), (0, no_of_zeros)), 'edge')  # 'constant', constant_values=(0))
        return x_padded

    def fit_func(self, x, y):
        img = x.reshape((self.image_dim, self.image_dim, 1), order='F')
        if self.convert_rgb:
            img = self.to_rgb(img)
        img = np.expand_dims(img, axis=0)
        if len(self.imgs) == 0:
            self.imgs = img
            self.labels = y
        else:
            self.imgs = np.append(self.imgs, img, axis=0)
            self.labels = np.append(self.labels, y, axis=0)
        return self

    def to_image(self, x, y):
        if self.randomize_features:
            x = self.permute(x, self.new_order)
        if len(self.indices) > 0:
            # print(X[:10,:10])
            x = self.permute(x, self.indices)
            # print(X[:10,:10])

        x_padded = self.pad(x)
        for row, row_label in zip(x_padded, y):
            self.fit_func(row, row_label)
        return

    def normalize(self, x_train):
        # self.imgs = (self.imgs - self.means) / (self.stds + 10 ** (-7))
        # self.imgs = (self.imgs - self.means) / (self.stds + 10 ** (-7))
        # X2.mean()

        return scale(x_train, 0, 255)

    def fit(self, x, y):
        # pad vector with zeros to get the appropriate length
        # X = scale(X, 0, 255)
        # X /= 255
        self.to_image(x, y)

        # calculate statistics
        self.calc_stats()

        # normalize
        # self.normalize()
        return

    def transform(self, x, y=None):
        self.to_image(x)
        return

