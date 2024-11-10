import torch
import torch.nn as nn

class PatternRecognizer(nn.Module):
    def __init__(self, num_patterns=10):
        super(PatternRecognizer, self).__init__()
        self.classifier = nn.Linear(128, num_patterns)

    def forward(self, encoded_grid):
        return torch.softmax(self.classifier(encoded_grid), dim=1)
