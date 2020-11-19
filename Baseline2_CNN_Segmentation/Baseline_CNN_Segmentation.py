import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN Net class definition
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding = 1) # Pad with zeros
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.fc1 = nn.Linear(8 * 12 * 12, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 12 * 12)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

net = Net()