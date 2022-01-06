import argparse
import torch
import torch.nn as nn

from utils import getFunction

class ConvBlock(nn.Module):
    def __init__(self, 
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE = 3,
        STRIDE = 1 ,
        PADDING = 0, 
        ACTIVATE_FUNC = "relu",
        USE_BATCHNORM = True,
        TRANSPOSE = False):
            
        super().__init__()

        self.TRANSPOSE = TRANSPOSE
        if(not TRANSPOSE):
            self.conv = nn.Conv2d (
                IN_CHANNELS, 
                OUT_CHANNELS.
                KERNEL_SIZE,
                stride = STRIDE, 
                padding = PADDING, 
                bias = False
            )
            
            self.conv = nn.ConvTranspose2d(
                IN_CHANNELS, 
                OUT_CHANNELS.
                KERNEL_SIZE,
                stride = STRIDE, 
                padding = PADDING, 
                bias = False
            )
        
        self.f = getFunction("relu")
        
        self.bn = USE_BATCHNORM
        if USE_BATCHNORM: 
            self.bn = nn.BatchNorm2d(OUT_CHANNELS)
        

    def forward(self, x):
        out = self.conv(x)
        
        if self.USE_BATCHNORM:
            out = self.bn(out)
        out = self.f(out)

        return out