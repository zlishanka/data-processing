import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from models.model import Module
from models.data import SyntheticRegressionData
from models.train import Trainer
from utils.util import add_to_class

class LinearRegression(Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

@add_to_class(LinearRegression)
def forward(self, X):
    return self.net(X)

@add_to_class(LinearRegression)
def loss(self, y_hat, y):
    fn = nn.MSELoss()
    return fn(y_hat, y)


# Instantiate an SGD instance, we specify the parameters to optimize over, obtainable 
# from our model via self.parameters(), and the learning rate (self.lr) required 
# by our optimization algorithm. 
@add_to_class(LinearRegression)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)


if __name__ == '__main__':
    # Training
    model = LinearRegression(lr=0.03)
    data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    trainer = Trainer(max_epochs=3)
    trainer.fit(model, data)
    plt.show()

    # Compare the model parameters learned by training on finite data 
    # and the actual parameters that generated our dataset.
    @add_to_class(LinearRegression)  
    def get_w_b(self):
        return (self.net.weight.data, self.net.bias.data)

    w, b = model.get_w_b()

    print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
    print(f'error in estimating b: {data.b - b}')

