import numpy as np
from preprocessing import train_val_test_split #, pre_process
from structureddataclass import StructDataTransformer
#from classification_tf2 import Classifier #apply_training, apply_testing,
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, "C:\\Users\\dl25365\\download\\dus\\CMBS\\")
#sys.path.insert(0, "i:\\works\\dus\\CMBS\\")
sys.path.insert(0, "/Users/drorlederman/download/madelon")

from importlib import reload

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold
from benchmarks import run_benckmarks
#from pipeline import pipeline

weights = [1,2,3,4]
no_of_samples = 10000
#cols = list('ABCDEFGHJIKLMNOP')
cols = [str(i) for i in range(0,64)]
data_vect = pd.DataFrame(np.random.randn(no_of_samples, len(cols)), columns=cols)
y = weights[0] * data_vect['0'] + weights[1] * data_vect['1'] + weights[2]*data_vect['2'] + weights[3] * data_vect['3']
labels = pd.DataFrame(columns={'dec'})
labels['dec'] = (y > 0)
labels[labels['dec']==True] = 1
labels[labels['dec']==False] = 0
x_train_vect, x_val_vect, x_test_vect, y_train, y_val, y_test = train_val_test_split(data_vect.to_numpy(), labels.to_numpy())
print(x_train_vect.shape, x_val_vect.shape, x_test_vect.shape)
batch_size = 256
num_classes = 2
epochs = 50
max_rows = 1000
no_of_batches = 0
no_of_scrambles = 0
augment = False
convert_rgb = True
# input image dimensions
fv = StructDataTransformer()
fv.fit(data_vect.to_numpy(), labels.to_numpy())
img_rows, img_cols = fv.image_dim, fv.image_dim
x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(fv.imgs, labels.to_numpy())
fv.augment(x_train, y_train, no_of_batches, no_of_scrambles)
fv.combine_data(x_train, y_train)
x_train = fv.combined_training_imgs.copy()
y_train = fv.combined_training_labels.copy()
input_shape = (img_rows, img_cols, 1)
print(fv.combined_training_imgs.shape, fv.combined_training_labels.shape)

#x_train, y_train, x_val, y_val, x_test, y_test, input_shape = pre_process(x_train, y_train, x_val, y_val, x_test, y_test, num_classes, img_rows, img_cols)
#model, history = apply_training(x_train, y_train, x_val, y_val, batch_size, epochs, input_shape, num_classes, augment)
#classifier = Classifier(x_train, y_train, x_val, y_val, x_test, y_test, num_classes, img_rows, img_cols, input_shape, batch_size, epochs)
#classifier.pre_process() #x_train, y_train, x_val, y_val, x_test, y_test, num_classes, img_rows, img_cols)
#classifier.apply_training() #x_train, y_train, x_val, y_val, batch_size, epochs, input_shape, num_classes, augment)
#model, history = pipeline(x_train, y_train, x_val, y_val, x_test, y_test, num_classes, img_rows, img_cols, batch_size, epochs, augment)
#score = classifier.apply_testing()
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

run_benckmarks(x_train_vect,y_train, x_val_vect, y_val, x_test_vect, y_test)

