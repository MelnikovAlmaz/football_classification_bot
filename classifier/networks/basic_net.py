import torch.nn as nn
import torch.nn.functional as F


class BasicNet(nn.Module):
    """
    Classify photo to 6 basic classes:
        1. Blue team players (10 classes merged)
        2. Green goalkeeper - goal keeper of blue team
        3. Other
        4. Refferi (2 classes merged)
        5. White team
        6. Yellow goalkeeper - goal keeper of white team
    """
    def __init__(self):
        super(BasicNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 13 * 6, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x