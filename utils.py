import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from preprocessing import my_train_val_test_split
import os
import torch
from file_ops import read_images

def folder_definition(setup):

    if (os.name == 'nt'):
        if setup==1:
            folder = r"C:\\Users\\dl25365\\works\\structureddata\\images\\"
        else:
            folder = r"D:\\Dror\\works\\structureddata\\images\\"

        resultsFolder = folder + "\\results1\\"
        dataFolder = folder + "\\results\\"
    else:
        folder = r"/Users/drorlederman/Documents/works/structureddata/images/"
        resultsFolder = folder + "/results1/"
        dataFolder = folder + "\\results\\"
    return folder, resultsFolder, dataFolder

def scale(X, x_min, x_max, X_min=[], X_max=[]):
    if (len(X_min) == 0):
        X_min = X.min(axis=0)
    if (len(X_max) == 0):
        X_max = X.max(axis=0)
    nom = (X - X_min) * (x_max - x_min)
    denom = X_max - X_min
    denom[denom == 0] = 1
    return (x_min + nom / denom), X_min, X_max


def scramble_image(img):
    no_rows, no_cols, no_of_channels = img.shape[0], img.shape[1], img.shape[2]
    num_pix = no_rows * no_cols * no_of_channels
    img_flat = img.reshape(num_pix, -1)
    perm_vect = np.random.permutation(num_pix)
    return img_flat[perm_vect].reshape(no_rows, no_cols, no_of_channels)


def to_rgb(img):
    return np.repeat(img[..., np.newaxis], 3, -1)

def define_transforms(data_mn, data_std):

    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(self.img_rows, self.img_cols, 1),
            #transforms.RandomHorizontalFlip(),
            #transforms.Resize((self.img_rows, self.img_cols, 1)),
            transforms.ToTensor(),
            #transforms.Normalize(data_mn, data_std)
        ]),
        'val': transforms.Compose([
            #transforms.Resize((self.img_rows, self.img_cols, 1)), #data = data.reshape(data.shape[0], 1, self.img_rows, self.img_cols)
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(data_mn, data_std)
        ]),
        'test': transforms.Compose([
            #transforms.Resize((self.img_rows, self.img_cols, 1)),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(data_mn, data_std)
        ]),
    }

    return data_transforms


class PlainDataSet(Dataset):
    def __init__(self, file_name, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.landmarks_frame = pd.read_csv(csv_file)
        #self.data = np.array([1.0] * 1000)
        self.data, self.labels = read_images(file_name)
        #self.root_dir = root_dir
        self.transform = transform

        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #img_name = os.path.join(self.root_dir,
        #                        self.data.iloc[idx, 0])
        #image = io.imread(img_name)
        image = []
        data = self.data[idx, :]
        labels = self.labels[idx, :]
        #data = np.array([self.data])
        #data = data.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'landmarks': data}

        if self.transform:
            data = self.transform(data)

        labels = torch.Tensor(list(labels))

        return data, labels

