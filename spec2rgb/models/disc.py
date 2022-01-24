from torch import nn

import torch.nn as nn
import torch


class PatchGAN(nn.Module):
    
    def __init__(self, input_channels):
        super().__init__()
        self.d1 = nn.Conv2d(input_channels, 64, stride=1, padding=1)
        self.d2 = nn.DownSampleConv(64, 128, stride=2, padding=2)
        self.d3 = nn.DownSampleConv(128, 256, stride=2, padding=1)
        self.d4 = nn.DownSampleConv(256, 512)
        self.final = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        xn = self.final(x3)
        return xn
    
    def _weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)