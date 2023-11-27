import numpy as np 
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
import torch

# A derivative is the rate of change in a function with respect to changes in its arguments.
# sum rule f(x) + g(x), product rule f(x)*g(x), Quotient rule f(x)/g(x)
 

# f'(x) = 6x - 4, x = 1 f'(x) = 2
def f(x):
    return 3 * x ** 2 - 4 * x

for h in 10.0**np.arange(-1,-6,-1):
    print(f'h={h:.5f}, f^(x)={(f(1+h)-f(1))/h}')




# # Partial Derivatives  

#  - In deep learning, functions often depend on many variables. Thus, we need to extend the ideas of differentiation to these multivariate functions.
#  - The partial derivative of function with respect to its ith parameter
#  - concatenate partial derivatives of a multivariate function with respect to all its variables to obtain a vector that is called the gradient of the function
#  - gradients are useful for designing optimization algorithms in deep learning.

# Chain Rule (composite functions)

# - y = f(g(x)) , both y = f(u) and u = g(x) are differentiable, functions g1,g2,...,gm, variabls x1,x2,...,xn
# - evalutate gradients dy/dx involves computing vector–matrix product A * gradientOfU * y
# - A is nxm matrix that contains the derivative of vector u with respect to vector x

# Automatic Differentiation (autograd)

# - all modern deep learning frameworks take this work off our plates by offering automatic differentiation
# - As we pass data through each successive function, the framework builds a computational graph that tracks how each value depends on others. 
# - To calculate derivatives, automatic differentiation works backwards through this graph applying the chain rule.
# - The computational algorithm for applying the chain rule in this fashion is called backpropagation.
# - backpropagate simply means to trace through the computational graph, filling in the partial derivatives with respect to each parameter.

# example to differentiate f(x) = 2 * xT *x with respect to vector x

x = torch.arange(4.0)
x.requires_grad_(True)
x.grad
print(f'x vector = {x}, gradient of x = {x.grad}')

y = 2 * torch.dot(x,x)
print(f'y=2*xT*x={y}')

# take the gradient of y with respect to x

y.backward()
print(f'x.grad={x.grad}')

# reset gradient 
x.grad.zero_()
y = x.sum()
y.backward()
print(f'y=x.sum={y}, x.grad={x.grad}')


# Jacobin matrix, dy/dx 
# When y is a vector, the most natural representation of the derivative of y with respect to a vector x is 
# a matrix called the Jacobian that contains the partial derivatives of each component of y with respect to each component of x.
# Likewise, for higher-order y and x, the result of differentiation could be an even higher-order tensor.

# For example, we often have a vector representing the value of our loss function calculated separately for 
# each example among a batch of training examples
# Here, we just want to sum up the gradients computed individually for each example.

# https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29

x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
print(f'y=x*x={y}, x.grad={x.grad}')

# Detaching Computation
# Sometimes, we wish to move some calculations outside of the recorded computational graph.
# suppose we have z = x * y and y = x * x but we want to focus on the direct influence of x on z rather than the 
# influence conveyed via y

# In this case, we can create a new variable u that takes the same value as y but whose provenance (how it was created) 
# has been wiped out. Thus u has no ancestors in the graph and gradients do not flow through u to x. 


# (i) attach gradients to those variables with respect to which we desire derivatives; 
# (ii) record the computation of the target value; 
# (iii) execute the backpropagation function; and 
# (iv) access the resulting gradient.

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

# x.grad == u
z.sum().backward()
print(f"x.grad={x.grad}, u={u}")

x.grad.zero_()
y.sum().backward()

# x.grad = 2 * x
print(f"x.grad={x.grad}, u={u}")

# building the computational graph of a function, output depends on input of a
# we can still calculate the gradient of the resulting variable. 
# f(a) is a linear function of a with piecewise defined scale

# Dynamic control flow is very common in deep learning. For instance, when processing text, 
# the computational graph depends on the length of the input. 

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

# a.grad == d / a 
print(f'd={d}, a={a}')
print(f'input a={a}, a.grad={a.grad}')