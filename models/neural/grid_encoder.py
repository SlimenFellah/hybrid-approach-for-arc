import torch
import torch.nn as nn

class GridEncoder(nn.Module):
    def __init__(self, input_size=(30, 30), num_features=128):
        super(GridEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * input_size[0] * input_size[1], num_features)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.fc(x)
        return x
