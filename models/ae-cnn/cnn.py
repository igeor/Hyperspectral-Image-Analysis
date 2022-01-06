import argparse
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Convolutional Neural Network')
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64,
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=10, 
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, 
                    help='learning rate (default: 0.01)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')


class Net(nn.Module):
    def __init__(self, 
                IN_CHANNELS = 2048,
                OUT_CHANNELS = 3,
                N_CONVBLOCKS = 4,
                KERNEL_SIZE = 3,
                STRIDE = 1 ):
        
        super(Net, self).__init__()
        x= 1
        
        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)