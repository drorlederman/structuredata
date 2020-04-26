from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Input, Dense

from preprocessing import scale
import numpy as np

class Classifier():
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, num_classes, img_rows, img_cols, input_shape, batch_size=256, epochs=100):
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
        return

    def pre_process(self):
        if K.image_data_format() == 'channels_first':
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
                x_train[i, :] = preprocess_input(row)
                i = i + 1
            i = 0
            for row in x_val:
                x_val[i, :] = preprocess_input(row)
                i = i + 1
            for row in x_test:
                x_test[i, :] = preprocess_input(row)
                i = i + 1
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_val = keras.utils.to_categorical(self.y_val, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

        return #x_train, y_train, x_val, y_val, x_test, y_test, input_shape

    def apply_training(self): #, x_train, y_train, x_test, y_test, batch_size, epochs, input_shape, num_classes, augment=False):
        l2_reg = 0.0001
        l1_reg = 0.0001
        sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        adam = optimizers.Adam(lr=5e-3, epsilon=1e-8, beta_1=.9, beta_2=.999)
        # augmentation
        # datagen = augmentation(x_train, y_train)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))  # ,
        # kernel_regularizer=regularizers.l2(l2_reg),
        # activity_regularizer=regularizers.l1(l1_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation='sigmoid'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=adam,
                      metrics=['accuracy'])
        # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, augment)
        # print(input_shape, x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        # if augment:
        #    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
        #              verbose=1,
        #              steps_per_epoch=math.ceil(x_train.shape[0]/ batch_size),
        #              validation_data=(x_test, y_test))
        # else:
        self.model = model
        self.history = self.model.fit(self.x_train, self.y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            callbacks=[es],
                            validation_data=(self.x_val, self.y_val))
        return


    def apply_training_resnet(self): #, x_train, y_train, x_test, y_test, batch_size, epochs, input_shape, num_classes,
                              #augment=False):
        l2_reg = 0.0001
        l1_reg = 0.0001
        sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        adam = optimizers.Adam(lr=1e-4, epsilon=1e-8, beta_1=.9, beta_2=.999)
        # augmentation
        # datagen = augmentation(x_train, y_train)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        # add new classifier layers
        flat1 = Flatten()(model.outputs)
        class1 = Dense(1024, activation='relu')(flat1)
        output = Dense(10, activation='softmax')(class1)
        # define new model
        self.model = Model(inputs=model.inputs, outputs=output)

        self.history = model.fit(self.x_train, self.y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            callbacks=[es],
                            validation_data=(self.x_val,self.y_val))
        return


    def apply_testing(self): #, model, x_test, y_test):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return score
