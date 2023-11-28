# Example to generate synthetic regression dataset
import random
import torch

from models.data import DataModule
from utils.util import add_to_class

class SyntheticRegressionData(DataModule):
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n,1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1,1))) + b + noise


data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
print('features:', data.X[0],'\nlabel:', data.y[0])

# add get_dataloader function's implementation
@add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        batch_indices = torch.tensor(indices[i: i+self.batch_size])
        # generate a batch (X,y) with batch_size
        yield self.X[batch_indices], self.y[batch_indices]

# xy_iterator = iter(data.train_dataloader())
# for i in range(100):
#     print(f"generate {i} round of batch data")
#     X,y = next(xy_iterator)
#     print('X shape:', X.shape, '\ny shape:', y.shape)


@add_to_class(DataModule)
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    """it is more efficient and has some added functionality."""
    tensors = tuple(a[indices] for a in tensors)
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(dataset, self.batch_size,
                                       shuffle=train)
@add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)

xy_iterator = iter(data.train_dataloader())
for i in range(len(data.train_dataloader())):
    X, y = next(xy_iterator)
    print('batch:', i, 'X shape:', X.shape, '\ny shape:', y.shape)