import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from utils.util import add_to_class
from models.data import FashionMNIST
from models.model import Classifier
from models.train import Trainer

class SoftmaxRegression(Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)

@add_to_class(Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    Y = Y.reshape((-1,))
    return F.cross_entropy(
        Y_hat, Y, reduction='mean' if averaged else 'none')

if __name__=='__main__':
    
    data = FashionMNIST(batch_size=256)
    model = SoftmaxRegression(num_outputs=10, lr=0.1)
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, data)
    plt.show()
