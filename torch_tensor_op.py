import torch 

# Tensor n dimensional array, resembles NumPy’s ndarray with few advanced features.                                 
# These properties make the tensor class suitable for deep learning.
#   (1) tensor class supports automatic differentiation.
#   (2) GPU is well-supported to accelerate the computation whereas NumPy only supports CPU computation

# https://pytorch.org/docs/stable/index.html 

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x.shape)
# total number of tensor nodes
print(x.numel())
print(x+y, x-y, x*y, x/y, x**y)
print(torch.exp(x))

# tensors initialized to contain all 0s or 1s.
torch.zeros((2, 3, 4))
torch.ones((2, 3, 4))

# generate random 12 numbers and reshape it to 3 by 4 matrix  
x = torch.randn(12).reshape(3,4)

# get maen, sum, norm (l2 normalization) 
print(x, x.mean(), x.sum(), x.norm())

X = torch.tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.]])
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# cat - cascade on dimension 0
print(X,Y, torch.cat((X, Y), dim=0))

# change to numpy 
A = X.numpy()
print(A)

# tensor get from numpy
B= torch.from_numpy(A)
print(type(A), type(B))

# transpose 
x = torch.randn(2,3)
xt = torch.t(x)
print(f'transpose of {x} is {xt}')

# dot product x,y are expected to 1D vector
x = torch.tensor([2,3])
y  = torch.tensor([2,1])
print(f'{x} dot {y} = {torch.dot(x,y)}')

# matrix and vector multiply
x = torch.tensor([[2,1],[1,2]])
y = torch.tensor([1,-1])
print(f'{x} x {y} = {torch.mv(x,y)}')


# matrix and matrix multiply
x = torch.ones(2,2)
y = 2 * torch.ones(2,2)
print(f'{x} x {y} = {torch.mm(x,y)}')

