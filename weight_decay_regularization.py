# how regularization solves overfitting 
#         - Overfitting means high variance 
#         - Limits the flexibility of the model 
# how to do the regularization 
#         - Constraint a model to simplify it (fewer drgrees of freedom) 
#             - L1/L2 regularization
#             - drop out
#             - early stopping 
#         - Add more information like data augmentation 

# The most common method for ensuring a small weight vector is to add its norm 
# as a penalty term to the problem of minimizing the loss.

import torch
from torch import nn
import matplotlib.pyplot as plt

from models.data import DataModule
from models.model import LinearRegressionScratch, LinearRegression
from models.train import Trainer

class Data(DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
    
def l2_penalty(w):
    return (w ** 2).sum() / 2

# Define a subclass of d2l.LinearRegressionScratch. The only change 
# here is that our loss now includes the penalty term.


class WeightDecayScratch(LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()

    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))

data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = Trainer(max_epochs=10)

def train_scratch(lambd):
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    print('L2 norm of w:', float(l2_penalty(model.w)))

# with lambd = 0, disabling weight decay. Note that we overfit badly, decreasing the training error 
# but not the validation error—a textbook case of overfitting.
train_scratch(0)
plt.show()

# Using Weight Decay
train_scratch(3)
plt.show()

# specify the weight decay hyperparameter directly through weight_decay when instantiating optimizer.
class WeightDecay(LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)

model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))

plt.show()
