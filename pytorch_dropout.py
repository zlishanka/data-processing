import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from utils.util import add_to_class, reshape
from models.model import Classifier
from models.data import FashionMNIST
from models.train import Trainer

class DropoutMLP(Classifier):

    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(),
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(),
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))

@add_to_class(DropoutMLP)
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = reshape(Y, (-1,))
    return F.cross_entropy(Y_hat, Y, reduction='mean' if averaged else 'none')

if __name__ == '__main__':
    hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
    model = DropoutMLP(**hparams)
    data = FashionMNIST(batch_size=256)
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, data)
    plt.show()
