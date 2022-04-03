import torch
import torch.nn as nn

class CancelOut(nn.Module):
    def __init__(self,inp, *input, **kwargs):
        super(CancelOut, self).__init__()
        self.weights = nn.Parameter(torch.zeros(inp))
        
    def forward(self, x):
        return (x * torch.sigmoid(self.weights.float()))