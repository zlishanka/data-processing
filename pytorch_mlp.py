from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from models.model import Classifier
from models.data import FashionMNIST
from models.train import Trainer

from utils.util import add_to_class, reshape

class MLP(Classifier):

    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))

@add_to_class(MLP)
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = reshape(Y, (-1,))
    return F.cross_entropy(Y_hat, Y, reduction='mean' if averaged else 'none')

if __name__ == '__main__':
    model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
    data = FashionMNIST(batch_size=256)
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, data)
    plt.show()
