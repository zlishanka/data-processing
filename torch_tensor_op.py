import torch 

# Tensor n dimensional array, resembles NumPy’s ndarray with few advanced features.                                 
# These properties make the tensor class suitable for deep learning.
#   (1) tensor class supports automatic differentiation.
#   (2) GPU is well-supported to accelerate the computation whereas NumPy only supports CPU computation

x = torch.arrange(12, dtype=torch.float32)
print(x)
print(x.shape)
print(x.numel())

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x+y, x-y, x*y, x/y, x**y
torch.exp(x)

