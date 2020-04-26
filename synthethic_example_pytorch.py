from classification_pytorch import Classifier #apply_training, apply_testing,
from preprocessing import train_val_test_split #, pre_process
import sys

sys.path.insert(0, "C:\\Users\\dl25365\\download\\dus\\CMBS\\")
sys.path.insert(0, "/Users/drorlederman/download/madelon")

from benchmarks import run_benckmarks
from utils import PlainDataSet, define_transforms
from calc_stat import calc_stat
from generate_images import generate_data, data2image
import torch
import argparse
from utils import folder_definition
#http://localhost:6006/
#params = {'batch_size': 64,
#          'shuffle': True,
#          'num_workers': 6}
batch_size = 256
num_classes = 2
epochs = 50
max_rows = 1000
no_of_batches = 0
no_of_scrambles = 0
augment = False
convert_rgb = True
one_hot = False
folder, resultsFolder, dataFolder = folder_definition(1)
train_file_name = folder + "train"
valid_file_name = folder + "valid"
test_file_name = folder + "test"
all_file_name = folder + "all"
#folder = "\\D:\\Dror\\works\\structureddata\\"
#train_file_name = "D:\\Dror\\works\\structureddata\\images\\train"
#valid_file_name = "D:\\Dror\\works\\structureddata\\images\\valid"
#test_file_name = "D:\\Dror\\works\\structureddata\\images\\test"
#all_file_name = "D:\\Dror\\works\\structureddata\\images\\all"
img_rows, img_cols = 8,8
input_shape = (img_rows, img_cols)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    return parser, args

#tensorboard --logdir=d:\works\dror\tensorboard\
#tensorboard --logdir=/Users/drorlederman/Documents/works/structureddata/images/
if __name__ == '__main__':
    parser, args = get_parser()
    train_loader = PlainDataSet(train_file_name)
    data_mn, data_std = calc_stat(train_loader)
    data_transforms = define_transforms(data_mn, data_std)
    train_set = PlainDataSet(train_file_name, transform=data_transforms['train'])
    valid_set = PlainDataSet(valid_file_name, transform=data_transforms['val'])
    test_set = PlainDataSet(test_file_name, transform=data_transforms['test'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

    classifier = Classifier(num_classes, img_rows, img_cols, input_shape, batch_size, epochs, data_mn, data_std)
    classifier.run(args, train_loader, valid_loader, test_loader)

    data_vect, labels = generate_data()
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(data_vect.to_numpy(), labels.to_numpy())
    run_benckmarks(x_train, y_train.astype('int'), x_val, y_val.astype('int'), x_test, y_test.astype('int'))
