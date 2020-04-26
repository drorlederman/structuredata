from __future__ import print_function
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from utils import scale
from tensorflow import keras
from keras import layers
from keras.models import Sequential, load_model
from keras import backend as k

import numpy as np

class Classifier():
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, num_classes, img_rows, img_cols, input_shape, batch_size=256, epochs=100, one_hot = False):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.num_classes = num_classes
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.one_hot = one_hot
        return

    def pre_process(self):
        #tf.keras.K.image_data_format() == 'channels_first'
        a = False
        if a:
            x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_val = self.x_val.reshape(self.x_val.shape[0], 1, self.img_rows, self.img_cols)
            x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.img_rows, self.img_cols)
            input_shape = (1, self.img_rows, self.img_cols)
        else:
            x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_val = self.x_val.reshape(self.x_val.shape[0], self.img_rows, self.img_cols, 1)
            x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, 1)
            input_shape = (self.img_rows, self.img_cols, 1)
        x_train = x_train.astype('float32')
        x_val = x_val.astype('float32')
        x_test = x_test.astype('float32')
        x_train, X_min, X_max = scale(x_train, 0, 255)
        x_val, _, _ = scale(x_val, 0, 255, X_min=X_min, X_max=X_max)
        x_test, _, _ = scale(x_test, 0, 255, X_min=X_min, X_max=X_max)
        x_train /= 255
        x_val /= 255
        x_test /= 255
        # convert class vectors to binary class matrices
        f = False
        if f:
            i = 0
            for row in x_train:
                x_train[i, :] = tf.keras.preprocess_input(row)
                i = i + 1
            i = 0
            for row in x_val:
                x_val[i, :] = tf.keras.preprocess_input(row)
                i = i + 1
            for row in x_test:
                x_test[i, :] = tf.keras.preprocess_input(row)
                i = i + 1
        if self.one_hot:
            self.y_train = tf.keras.utils.to_categorical(self.y_train, self.num_classes)
            self.y_val = tf.keras.utils.to_categorical(self.y_val, self.num_classes)
            self.y_test = tf.keras.utils.to_categorical(self.y_test, self.num_classes)

        return #x_train, y_train, x_val, y_val, x_test, y_test, input_shape

    def apply_training(self): #, x_train, y_train, x_test, y_test, batch_size, epochs, input_shape, num_classes, augment=False):
        l2_reg = 0.0001
        l1_reg = 0.0001
        sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        adam = tf.keras.optimizers.Adam(lr=5e-3, epsilon=1e-8, beta_1=.9, beta_2=.999)
        # augmentation
        # datagen = augmentation(x_train, y_train)
        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
            baseline=None, restore_best_weights=False
        )

        tf.compat.v1.disable_eager_execution()
        print(tf.compat.v1.get_default_graph())
        #sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        #tf.compat.v1.keras.backend.set_session(sess)


        inputs = keras.Input(shape=(8,8), name='digits')
        x = layers.Conv2D(32, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dense(64, activation='relu', name='dense_2')(x)
        outputs = layers.Dense(10, name='predictions')(x)
        model2 = keras.Model(inputs=inputs, outputs=outputs)


        model = Sequential([
            layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Dropout(0.4),
            layers.Flatten(),
            #tf.keras.layers.Dense(128, activation='relu'),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='sigmoid'),
        ])

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model2.compile(optimizer=adam,
                      loss=loss_fn,
                      metrics=['accuracy'])
        model.compile(optimizer=adam,
                      loss=loss_fn,
                      metrics=['accuracy'])

        self.model = model
        self.history = self.model.fit(self.x_train, self.y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            callbacks=[es],
                            validation_data=(self.x_val, self.y_val))
        return



    def apply_testing(self): #, model, x_test, y_test):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return score
