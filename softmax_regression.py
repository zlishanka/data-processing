import torch
import matplotlib.pyplot as plt


from utils.util import add_to_class
from models.model import Classifier
from models.train import Trainer
from models.data import FashionMNIST

#from d2l import torch as d2l

# X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# print(X.sum(0, keepdims=True)) # add row
# print(X.sum(1, keepdims=True)) # add col

# Computing the softmax requires three steps: 
#     (i) exponentiation of each term; 
#     (ii) a sum over each row to compute the normalization constant for each example; 
#     (iii) division of each row by its normalization constant, ensuring that the result sums to 1:

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here

# X = torch.rand((2, 5))
# X_prob = softmax(X)
# print(f'X_prob={X_prob}')
# print(f'X_prob.sum(1)={X_prob.sum(1)}')

# Flatten 28x28 pixel images and treat them as vectors of length 784. 
# dataset has 10 classes so output has dimension of 10
# weights constitute a 784x10 matrix plus a 1x10 row vector for the biases.

class SoftmaxRegressionScratch(Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]

@add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = X.reshape((-1, self.W.shape[0]))
    return softmax(torch.matmul(X, self.W) + self.b)

# cross-entropy loss function 

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()

@add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)

# Training, train the model with 10 epochs.

data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
plt.show()