import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            #nn.Linear(40000, 10000),
            #nn.ReLU(),
            nn.Linear(10000, 784),
            nn.ReLU(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.ReLU(),
            nn.Linear(784,10000),
            #nn.ReLU(),
            #nn.Linear(10000,40000),
            nn.Sigmoid()
        )

    def forward(self, x,batch_size):
        x = x.view(x.size()[0], -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(batch_size, 1, 100, 100)
        return x