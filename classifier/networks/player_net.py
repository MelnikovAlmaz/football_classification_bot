import torch.nn as nn
import torch.nn.functional as F


class PlayerNet(nn.Module):
    """
        Classify players photo to 10 player classes
    """
    def __init__(self):
        super(PlayerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.fc1 = nn.Linear(12 * 13 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 13 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
