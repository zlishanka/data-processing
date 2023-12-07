# Overfitting and Regularization
# approach to training them typically consists of two phases:
# (i) fit the training data; and
# (ii) estimate the generalization error (the true error on the underlying population) by
#      evaluating the model on holdout data.

# Difference between our fit on the training data and our fit on the test data is called the generalization gap
# and when this is large, we say that our models overfit to the training data
# any improvements in generalization error must come by way of regularization,
# either by reducing the complexity of the model class, or by applying a penalty

# Ways to reduce generalization gap
# (1) Choose among model architectures, reduce the generalization error further by making the model
#     even more expressive - adding layers, nodes, or training for a larger number of epochs
# (2) Choose a nonparametric model, i.e. the  K-nearest neighbor algorithm
# (3) early stopping, monitor validation error throughout traininghs of training, stop at certain epoch.
# (4) in classical regularization contexts, such as adding noise to model inputs, dropout

import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from utils.util import plot, reshape, add_to_class
from models.model import Classifier
from models.data import FashionMNIST
from models.train import Trainer

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

# X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
# print('dropout_p = 0:', dropout_layer(X, 0))
# print('dropout_p = 0.5:', dropout_layer(X, 0.5))
# print('dropout_p = 1:', dropout_layer(X, 1))

class DropoutMLPScratch(Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)

# Need to add loss function
@add_to_class(DropoutMLPScratch)
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = reshape(Y, (-1,))
    return F.cross_entropy(Y_hat, Y, reduction='mean' if averaged else 'none')

if __name__ == '__main__':
    hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
    model = DropoutMLPScratch(**hparams)
    data = FashionMNIST(batch_size=256)
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, data)
    plt.show()
