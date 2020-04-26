from preprocessing import train_val_test_split, pre_process, my_train_val_test_split
from structureddataclass import StructDataTransformer
import sys

sys.path.insert(0, "/Users/drorlederman/download/madelon")

import pandas as pd
import numpy as np
import os
from file_ops import save_images

def generate_data():
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

    return data_vect, labels

def data2image(data_vect, labels, num_classes, one_hot):
    # input image dimensions
    fv = StructDataTransformer()
    fv.fit(data_vect.to_numpy(), labels.to_numpy())
    img_rows, img_cols = fv.image_dim, fv.image_dim

    # pre-processing
    fv.imgs, labels, input_shape, data_mn, data_std = pre_process(fv.imgs, img_rows, img_cols, labels, num_classes, one_hot)

    return fv, labels, input_shape, data_mn, data_std

if __name__ == '__main__':
    if 0:
        folder = "D:\\Dror\\works\\structureddata\\images\\"
    else:
        folder = "/drorlederman/Documents/work/structureddata/"
    train_file_name = folder + "train"
    valid_file_name = folder + "valid"
    test_file_name = folder + "test"
    all_file_name = folder + "all"
    #valid_file_name = "D:\\Dror\\works\\structureddata\\images\\valid"
    #test_file_name = "D:\\Dror\\works\\structureddata\\images\\test"
    #all_file_name = "D:\\Dror\\works\\structureddata\\images\\all"
    #full_path = folder + str(labels.loc[i, 'dec']) + "\\"
    #file_name = full_path + str(i) + '.png'
    num_classes = 2
    one_hot = False
    data_vect, labels = generate_data()
    fv, labels, input_shape, data_mn, data_std = data2image(data_vect, labels, num_classes, one_hot)
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(fv.imgs, labels.to_numpy())

    save_images(all_file_name, fv.imgs, labels)
    save_images(train_file_name, x_train, y_train)
    save_images(valid_file_name, x_val, y_val)
    save_images(test_file_name, x_test, y_test)



