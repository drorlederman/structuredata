import keras
from sklearn.model_selection import train_test_split

def train_val_test_split(data_vect, labels, train_size=0.7, val_size=0.15, test_size=0.15):
    x_train_vect, x_test_vect, y_train, y_test = train_test_split(data_vect, labels, test_size=(val_size + test_size))
    x_val_vect, x_test_vect, y_val, y_test = train_test_split(x_test_vect, y_test,
                                                              test_size=test_size / (1 - train_size))
    return x_train_vect, x_val_vect, x_test_vect, y_train, y_val, y_test

