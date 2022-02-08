import torch
from torch import nn

class FCN4(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.m = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.m(x)