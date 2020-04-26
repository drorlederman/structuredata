import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold

def folder_definition(setup):

    if (os.name == 'nt'):
        if setup==1:
            folder = r"C:\\Users\\dl25365\\works\\madelon\\"
            folder = r"i:\\works\\jupyternotebooks\\madelon\\"
        else:
            folder = r"C:\\Users\\Dror\\Downloads\\madelon\\"
        resultsFolder = folder + "\\results1\\"
        dataFolder = folder + "\\results\\"
        image_path = folder + '\\data\\'
        folder_mark = '\\'
        augmented_image_path = folder + '\\augmented\\'
    else:
        folder = r"/Users/drorlederman/Documents/works/madelon/"
        resultsFolder = folder + "/results1/"
        dataFolder = folder + "/results/"
        image_path = folder + '/data/'
        augmented_image_path = folder + '/augmented/'
        folder_mark = '/'
    return folder, resultsFolder, dataFolder, image_path, augmented_image_path, folder_mark

setup = 1
folder, resultsFolder, dataFolder, image_path, augmented_image_path, folder_mark = folder_definition(setup)
data = pd.read_csv(folder+"madelon_train.data", header=None, delimiter=' ')
labels = pd.read_csv(folder+"madelon_train.labels", header=None)
data_valid = pd.read_csv(folder+"madelon_valid.data", header=None, delimiter=' ')
labels_valid = pd.read_csv(folder+"madelon_valid.labels", header=None)
#indices = np.load('indices.npy')
indices = []
data_merged = pd.concat([data, data_valid])
labels_merged = pd.concat([labels, labels_valid])

data = data_merged.copy().reset_index(drop=True)
labels = labels_merged.copy().reset_index(drop=True)
labels[labels == -1] = 0
data = data.dropna(axis=1)