import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

from utils.util import add_to_class, reshape
from models.data import FashionMNIST
from models.model import Classifier
from models.train import Trainer

class MLPScratch(Classifier):

    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X,a)

@add_to_class(MLPScratch)
def forward(self, X):
    X = X.reshape((-1, self.num_inputs))
    H = relu(torch.matmul(X, self.W1) + self.b1)
    return torch.matmul(H, self.W2) + self.b2

# Need to add loss function
@add_to_class(MLPScratch)
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = reshape(Y, (-1,))
    return F.cross_entropy(Y_hat, Y, reduction='mean' if averaged else 'none')

if __name__ == '__main__':
    model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
    data = FashionMNIST(batch_size=256)
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, data)
    plt.show()

