import torch
import torch.nn as nn
import numpy as np
from config import Config

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.attention = SelfAttention(128)
        
    def forward(self, x):
        # x shape: (batch_size, 1, height, width)
        features = self.conv_layers(x)
        attended_features = self.attention(features)
        return attended_features

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        q = self.query(x).view(B, -1, H * W).transpose(1, 2)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W).transpose(1, 2)
        
        attention = torch.softmax(torch.bmm(q, k) / np.sqrt(C), dim=2)
        out = torch.bmm(attention, v).transpose(1, 2).view(B, C, H, W)
        
        return out + x  # Skip connection
