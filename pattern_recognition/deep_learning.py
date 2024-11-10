import torch
import torch.nn as nn
from typing import Dict, Any

class PatternRecognitionModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.pattern_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 10)  # Number of pattern classes
        )
        
        self.transformation_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 20)  # Number of transformation types
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(x)
        
        return {
            'pattern_class': self.pattern_classifier(features),
            'transformation': self.transformation_predictor(features)
        }