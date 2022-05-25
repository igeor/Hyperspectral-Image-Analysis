import torch
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self, num_features, in_channels, num_targets, poolSize=5, kernel_size=5, stide=1, padding=1):

        super(ConvModel, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=1, 
                kernel_size=kernel_size, stride=stide, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(poolSize)
        )

        self.l1 = nn.Sequential(
            nn.Linear(num_features // poolSize, 100),
            nn.ReLU()
        )

        self.lOut = nn.Sequential(
            nn.Linear(100, num_targets),
            nn.ReLU()
        )

    def forward(self, x_in):
        x1 = self.conv1(x_in)
        x2 = self.l1(x1)
        xOut = self.lOut(x2)
        return xOut