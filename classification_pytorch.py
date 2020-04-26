from __future__ import print_function
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from utils import scale
from tensorflow import keras
#from keras import layers
#from keras.models import Sequential, load_model
#from keras import backend as k
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) #(n samples, channels, height, width)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 2)
        #self.fc1 = nn.Flatten()
        #self.fc2 = nn.Softmax(dim=None)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        #output = F.softmax(x, dim=1)
        output = F.log_softmax(x, dim=1)
        return output


class Classifier():
    def __init__(self, args, num_classes, img_rows, img_cols, input_shape, batch_size=256, epochs=100, one_hot = False, data_mn = None, data_std = None):
        #self.train_loader = train_loader
        #self.val_loader = val_loader
        #self.test_loader = test_loader
        self.num_classes = num_classes
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.one_hot = one_hot
        self.data_mn = data_mn
        self.data_std = data_std
        self.writer = SummaryWriter('d:\\dror\\works\\tensorboard\\')

        return


    """
    def pre_process(self, device):
        def perform_reshape(data, a):
            if a:
                data = data.reshape(data.shape[0], 1, self.img_rows, self.img_cols)
                input_shape = (1, self.img_rows, self.img_cols)
            else:
                data = data.reshape(data.shape[0], self.img_rows, self.img_cols, 1)
                input_shape = (self.img_rows, self.img_cols, 1)

            data = data.astype('float32')

            return data, input_shape

        #tf.keras.K.image_data_format() == 'channels_first'
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            data, input_shape = perform_reshape(data, False)
            data, X_min, X_max = scale(data, 0, 255)
            data /= 255
        for batch_idx, (data, target) in enumerate(self.val_loader):
            data, target = data.to(device), target.to(device)
            data, input_shape = perform_reshape(data, False)
            data, _, _ = scale(data, 0, 255, X_min=X_min, X_max=X_max)
            data /= 255
        for batch_idx, (data, target) in enumerate(self.test_loader):
            data, target = data.to(device), target.to(device)
            data, input_shape = perform_reshape(data, False)
            data, _, _ = scale(data, 0, 255, X_min=X_min, X_max=X_max)
            data /= 255

        #if self.one_hot:
        #    self.y_train = tf.keras.utils.to_categorical(self.y_train, self.num_classes)
        #    self.y_val = tf.keras.utils.to_categorical(self.y_val, self.num_classes)
        #    self.y_test = tf.keras.utils.to_categorical(self.y_test, self.num_classes)


        return
    """
    def loss_criterion(self, output, target):
        loss = F.nll_loss(output, target)
        # loss = nn.CrossEntropyLoss()
        # loss = nn.CrossEntropyLoss(output, target)
        # loss = F.cross_entropy(output, target)
        return loss

    def train(self, args, model, device, train_loader, optimizer, epoch, loss_vect=None):

        running_loss = 0
        #summary_writer = tf.train.SummaryWriter('/tensorflow/logdir', sess.graph_def)
        for batch_idx, (data, target) in enumerate(train_loader):
            #target = target.to(device=device, dtype=torch.int64)
            model.train()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            #label = output.argmax(dim=1, keepdim=True).item()
            #target = torch.tensor(target, dtype=torch.int64, device=device)
            output = output.to(device=device) #, dtype=torch.int64)
            output = torch.squeeze(output)
            target = torch.squeeze(target)
            target = torch.tensor(target, dtype=torch.long, device=device)
            loss = self.loss_criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                niter = epoch * len(train_loader) + batch_idx
                #self.writer.add_scalar('Train/Loss', loss.item(), niter)

        running_loss /= len(train_loader.dataset)
        self.writer.add_scalar('Train/Loss', running_loss, epoch)
        if loss_vect:
            loss_vect.append(running_loss)

        return loss_vect

    def eval(self, args, model, device, eval_loader, epoch, loss_vect=None):
        model.eval()
        valid_loss = 0
        eval_loss = 0
        correct = 0
        #writer = SummaryWriter('runs')
        #writer = SummaryWriter('d:\\dror\\works\\tensorboards\\')
        with torch.no_grad():
            #for data, target in eval_loader:
            for batch_idx, (data, target) in enumerate(eval_loader):
                #target = target.to(device=device, dtype=torch.int64)
                data, target = data.to(device), target.to(device)
                output = model(data)
                #target = target.long()
                #target = target.type(torch.LongTensor)
                target = torch.squeeze(target)
                target = torch.tensor(target, dtype=torch.long, device=device)
                #loss = F.nll_loss(output, target) #, reduction='sum')  # sum up batch loss
                loss = self.loss_criterion(output, target)
                eval_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                valid_loss += loss.item() * data.size(0)
                niter = epoch * len(eval_loader) + batch_idx
                #if (epoch > 0): self.writer.add_scalar('Valid/Loss', loss.item(), niter)

        eval_loss /= len(eval_loader.dataset)
        if loss_vect:
            loss_vect.append(valid_loss)

        print('\n Tested set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            eval_loss, correct, len(eval_loader.dataset),
            100. * correct / len(eval_loader.dataset)))


        return eval_loss, correct, loss_vect

    def run(self, args, train_loader, valid_loader, test_loader):
        # args = None
        use_cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)

        device = torch.device("cuda" if use_cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        train_losses, valid_losses = [], []
        for epoch in range(1, args.epochs + 1):
            train_losses = self.train(args, model, device, train_loader, optimizer, epoch, loss_vect=train_losses)
            eval_loss, correct, valid_losses = self.eval(args, model, device, valid_loader, epoch, loss_vect=valid_losses)
            scheduler.step()

        self.eval(args, model, device, test_loader, 0)
        self.writer.close()


        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")

        plt.plot(train_losses, label = 'Train loss')
        plt.plot(valid_losses, label = 'Valid loss')
        plt.legend()

        return
