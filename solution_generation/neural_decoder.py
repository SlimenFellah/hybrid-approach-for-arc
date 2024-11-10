import torch
import torch.nn as nn
import torch.nn.functional as F

class GridEncoder(nn.Module):
    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        return x

class GridDecoder(nn.Module):
    def __init__(self, output_channels: int = 1):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv3 = nn.Conv2d(32, output_channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class NeuralDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GridEncoder()
        self.decoder = GridDecoder()
        self.attention = nn.MultiheadAttention(128, 8)
        
    def forward(self, input_grid: torch.Tensor) -> torch.Tensor:
        # Add batch and channel dimensions if necessary
        if len(input_grid.shape) == 2:
            input_grid = input_grid.unsqueeze(0).unsqueeze(0)
        elif len(input_grid.shape) == 3:
            input_grid = input_grid.unsqueeze(0)
            
        # Encode
        encoded = self.encoder(input_grid)
        
        # Reshape for attention
        b, c, h, w = encoded.shape
        encoded_flat = encoded.view(b, c, h*w).permute(2, 0, 1)
        
        # Apply attention
        attended, _ = self.attention(encoded_flat, encoded_flat, encoded_flat)
        attended = attended.permute(1, 2, 0).view(b, c, h, w)
        
        # Decode
        output = self.decoder(attended)
        
        return output.squeeze()