import torch
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self, in_channels, num_targets, mid=8):

        super(ConvModel, self).__init__()
        
        self.l1 = nn.Sequential(
            nn.Conv1d(in_channels, mid, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.l2 = nn.Sequential(
            nn.Conv1d(mid, mid * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.l3 = nn.Sequential(
            nn.Conv1d(mid * 2, mid * 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.l4 = nn.Sequential(
            nn.Conv1d(mid * 4, mid * 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.mlp = nn.Sequential(
            nn.Linear((mid * 8) * (1948 // 2 // 2 // 2 // 2), 100),
            nn.ReLU(),
            nn.Linear(100, num_targets)
        )

    def forward(self, x_in):

        x1 = self.l1(x_in)#; print(x1.shape)
        x2 = self.l2(x1)#; print(x2.shape)
        x3 = self.l3(x2)#; print(x3.shape)
        x4 = self.l4(x3)#; print(x4.shape)
        x_flat = torch.flatten(x4, start_dim=1, end_dim=- 1)
        x_out = self.mlp(x_flat)
        return x_out