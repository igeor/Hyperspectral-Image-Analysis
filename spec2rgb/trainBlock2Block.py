import torch
from torch.utils import data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data import DataLoader
import argparse

from datasets import PanagiaDataset
from models import ConvNN

parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batchsize", default=64)
parser.add_argument("-ep", "--epochs", default=100)
parser.add_argument("-lr", "--lrate", default=0.0002)
parser.add_argument("-omn", "--outmodelname", default="ConvNN")
parser.add_argument("-dev", "--device", default="cpu")

args = parser.parse_args()

imageDataset = PanagiaDataset("panagia.h5")
dataLoader = DataLoader(dataset=imageDataset, batch_size=args.batchsize, shuffle=True)

model = ConvNN(in_channels=2048, out_channels=3).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
loss_fn = nn.L1Loss()

    
for epoch in range(int(args.epochs)):
    
    total_loss = 0

    for i, x_in in enumerate(dataLoader):
        
        y_out = model(x_in.to(args.device))
        y_real = torch.tensor(torch.rand(y_out.shape)).to(args.device)
        print(y_out.shape, y_real.shape)
        loss = loss_fn(y_out, y_real)
        total_loss += loss * 100

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # if(total_loss < min_total_loss):
    #     torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': autoencoder.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': total_loss,
    #             }, PATH)
    #     min_total_loss = total_loss
    

    # if(epoch % 10 == 0 and epoch > 0):
    #     print(epoch, total_loss / len(dataLoader))
