# Fashion-MNIST dataset (Xiao et al.) which was released in 2017. It contains images 
# of 10 categories of clothing at 28x28 pixels resolution.

# Fashion-MNIST consists of images from 10 categories, each represented by 
# 6000 images in the training dataset and by 1000 in the test dataset. A test dataset 
# is used for evaluating model performance (it must not be used for training). 
# Consequently the training set and the test set contain 60,000 and 10,000 images, respectively.

import time
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from utils.util import  add_to_class, show_images
from models.data import DataModule
from models.model import Module

class FashionMNIST(DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)



# converts between numeric labels and their names
@add_to_class(FashionMNIST)
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]

# Reading a Minibatch
@add_to_class(FashionMNIST)
def get_dataloader(self, train):
    """use the built-in data iterator"""
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)


# X, y = next(iter(data.train_dataloader()))
# print(X.shape, X.dtype, y.shape, y.dtype)

# tic = time.time()
# for X, y in data.train_dataloader():
#     continue
# print(f'{time.time() - tic:.2f} sec')
# def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
#     """Plot a list of images."""
#     raise NotImplementedError

@add_to_class(FashionMNIST)
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    show_images(X.squeeze(1), nrows, ncols, titles=labels)
class Classifier(Module):  #@save
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

@add_to_class(Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)

@add_to_class(Classifier)
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    preds = Y_hat.argmax(axis=1).type(Y.dtype)
    compare = (preds == Y.reshape(-1)).type(torch.float32)
    return compare.mean() if averaged else compare

if __name__=='__main__':
    data = FashionMNIST(resize=(32, 32))
    print(f'Size of training set: {len(data.train)}, Size of validation set: {len(data.val)}')
    batch = next(iter(data.val_dataloader()))
    data.visualize(batch)
    plt.show()

