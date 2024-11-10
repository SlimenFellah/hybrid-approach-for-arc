import torch
import torch.nn as nn

class GridEncoder(nn.Module):
    def __init__(self, input_size=(30, 30), num_features=128):
        super(GridEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Forward pass a dummy input to determine the flattened size after the conv layers
        self.flattened_size = self._get_flattened_size(input_size)
        
        self.fc = nn.Linear(self.flattened_size, num_features)

    def _get_flattened_size(self, input_size):
        """Passes a dummy input through the conv layers to determine the flattened size."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_size)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            return x.numel()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        print('After conv1:', x.shape)
        x = torch.relu(self.conv2(x))
        print('After conv2:', x.shape)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        print('After flattening:', x.shape)
        return self.fc(x)

