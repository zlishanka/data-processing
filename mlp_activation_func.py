import torch
import matplotlib.pyplot as plt

from utils.util import plot

# Show the Activation function ReLU(x)
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
plt.show()

# Derivative of ReLU
y.backward(torch.ones_like(x), retain_graph=True)
plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5,2.5))
plt.show()

# The reason for using ReLU is that its derivatives are particularly
# well behaved: either they vanish or they just let the argument through.

y = torch.sigmoid(x)
plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5,2.5))
plt.show()

# Clear out previous gradients
x.grad.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
plt.show()

y = torch.tanh(x)
plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
plt.show()

x.grad.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
plt.show()
