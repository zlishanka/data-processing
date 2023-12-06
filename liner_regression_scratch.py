import torch
import matplotlib.pyplot as plt

from models import model
from utils.util import add_to_class
from utils.util import HyperParameters, SGD
from models.train import Trainer
from models.data import SyntheticRegressionData, DataModule

class LinearRegressionScratch(model.Module):
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        
@add_to_class(LinearRegressionScratch)
def forward(self, X):
    return torch.matmul(X, self.w) + self.b

@add_to_class(LinearRegressionScratch)
def loss(self, y_hat, y):
    l = (y_hat - y)**2/2
    return l.mean()

@add_to_class(LinearRegressionScratch)
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr)

# Training 
# all of the parts in place (parameters, loss function, model, and optimizer)
# implement the main training loop.

# In each epoch, we iterate through the entire training dataset, passing once through every example

# In each iteration, 
# (1) grab a minibatch of training examples, 
# (2) compute its loss through the model’s training_step method
# (3) compute the gradients with respect to each parameter
# (4) call the optimization algorithm to update the model parameters.

@add_to_class(Trainer)
def prepare_batch(self, batch):
    return batch

@add_to_class(Trainer)
def fit_epoch(self):
    self.model.train()
    for batch in self.train_dataloader:
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1


@add_to_class(DataModule)
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    """it is more efficient and has some added functionality."""
    tensors = tuple(a[indices] for a in tensors)
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(dataset, self.batch_size,
                                       shuffle=train)
    
# define dataloader 
@add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)


if __name__=='__main__':
    model = LinearRegressionScratch(2, lr=0.03)
    data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    trainer = Trainer(max_epochs=3)
    trainer.fit(model, data)
    plt.show()
