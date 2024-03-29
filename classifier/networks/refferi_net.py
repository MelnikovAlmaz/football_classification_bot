import torch.nn as nn
import torch.nn.functional as F


class RefferiNet(nn.Module):
    """
        Classify refferi photo to 2 classes:
            1. Main refferi
            2. Side refferi
        """
    def __init__(self):
        super(RefferiNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 3)
        self.fc1 = nn.Linear(10 * 13 * 6, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 13 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
