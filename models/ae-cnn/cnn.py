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
                PADDING = 0):
        
        super(Net, self).__init__()
        self.IN_CHANNELS = IN_CHANNELS
        self.OUT_CHANNELS = OUT_CHANNELS
        self.N_CONVBLOCKS = N_CONVBLOCKS
        self.KERNEL_SIZE = KERNEL_SIZE
        self.STRIDE = STRIDE
        self.PADDING = PADDING

        self.convBlocks = list()
        
        # Building encoder
        for i in range(N_CONVBLOCKS):
            self.e1 = 1        
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)