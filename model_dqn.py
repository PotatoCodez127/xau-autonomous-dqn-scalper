# model_dqn.py
import torch
import torch.nn as nn

class TradingBrain(nn.Module):
    def __init__(self, input_dim=2, output_dim=3):
        super(TradingBrain, self).__init__()
        # Input: [Current Price, Current Position]
        # Output: [Q(Hold), Q(Buy), Q(Sell)]
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)