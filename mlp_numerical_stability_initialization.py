import torch
import matplotlib.pyplot as plt

from utils.util import plot

# Vanishing and Exploding Gradients
# Parameter updates that are either
# (i) excessively large, destroying our model (the exploding gradient problem); or
# (ii) excessively small (the vanishing gradient problem), rendering learning impossible as
#      parameters hardly move on each update.

# The following example shows that the sigmoid’s gradient vanishes both when
# its inputs are large and when they are small.

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))
plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
plt.show()

# 100 Gaussian random matrices and multiply them with some initial matrix.
# For the scale that we picked (the choice of the variance = 1, the matrix product explodes.
# have no chance of getting a gradient descent optimizer to converge.

# ReLU activation functions mitigate the vanishing gradient problem.

M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))
print('after multiplying 100 matrices\n', M)

# Default Initialization - a normal distribution to initialize the values of weights
# Xavier initialization - samples weights from a Gaussian distribution with zero mean and
# variance (2/(num_input + num_output))
