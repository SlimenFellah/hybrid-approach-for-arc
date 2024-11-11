import numpy as np
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from config import Config

class StrategySelector(nn.Module):
    def __init__(self, feature_dim: int = 256, num_strategies: int = 5):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_strategies)
        )
        
        self.strategy_descriptions = {
            0: "direct_mapping",
            1: "pattern_completion",
            2: "geometric_transformation",
            3: "color_transformation",
            4: "composite_transformation"
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.feature_encoder(x), dim=-1)
    
    def select_strategy(self, features: torch.Tensor) -> Tuple[int, str, float]:
        with torch.no_grad():
            probabilities = self(features)
            strategy_idx = torch.argmax(probabilities).item()
            confidence = probabilities[strategy_idx].item()
            return strategy_idx, self.strategy_descriptions[strategy_idx], confidence
