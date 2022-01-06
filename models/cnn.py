import argparse
import torch
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