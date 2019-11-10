import torch.nn as nn
import torch.nn.functional as F


class RefferiNet(nn.Module):
    def __init__(self):
        super(RefferiNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 14 * 29, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6 * 14 * 29)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
