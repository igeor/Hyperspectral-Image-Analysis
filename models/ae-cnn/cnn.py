import argparse
import torch.nn as nn
import torch.nn.functional as F

from convblock import ConvBlock

parser = argparse.ArgumentParser(description='Convolutional Neural Network')
parser.add_argument('--IN_CHANNELS', type=int, default=64,
                    help='num of input channels (default: 2048)')
parser.add_argument('--OUT_CHANNELS', type=int, default=64,
                    help='num of output channels (default: 3)')
parser.add_argument('--N_CONVBLOCKS', type=int, default=64,
                    help='num of convblocks of encoder and decoder (default: 4)')
parser.add_argument('--KERNEL_SIZE', type=int, default=64,
                    help='size of kernel of convolutional operation (default: 3)')
parser.add_argument('--STRIDE', type=int, default=64,
                    help='stride of convolutional operation (default: 1)')
parser.add_argument('--PADDING', type=int, default=64,
                    help='padding of convolutional operation (default: 0)')

class Net(nn.Module):
    def __init__(self, 
                IN_CHANNELS = 2048,
                OUT_CHANNELS = 3,
                N_CONVBLOCKS = 4,
                KERNEL_SIZE = 3,
                STRIDE = 1 ,
                PADDING = 0,
                ACTIVATE_FUNC = "relu",
                USE_BATCHNORM = True):
        
        super(Net, self).__init__()
        self.IN_CHANNELS = IN_CHANNELS
        self.OUT_CHANNELS = OUT_CHANNELS
        self.N_CONVBLOCKS = N_CONVBLOCKS
        self.KERNEL_SIZE = KERNEL_SIZE
        self.STRIDE = STRIDE
        self.PADDING = PADDING
        self.ACTIVATE_FUNC = ACTIVATE_FUNC
        self.USE_BATCHNORM = USE_BATCHNORM

        self.convBlocks = list()
        
        # Building encoder
        for i in range(N_CONVBLOCKS):
            self.convBlocks.append(
                ConvBlock( self.IN_CHANNELS, 
                    self.IN_CHANNELS // 2,
                    self.KERNEL_SIZE,
                    self.STRIDE,
                    self.PADDING,
                    self.ACTIVATE_FUNC,
                    self.USE_BATCHNORM))
            
        # Building encoder
        for i in range(N_CONVBLOCKS):
            self.convBlocks.append(
                ConvBlock( self.IN_CHANNELS, 
                    self.IN_CHANNELS // 2,
                    self.KERNEL_SIZE,
                    self.STRIDE,
                    self.PADDING,
                    self.ACTIVATE_FUNC,
                    self.USE_BATCHNORM,
                    TRANSPOSE=True))
            