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

model = ConvNN().to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
loss_fn = nn.L1Loss()

for i in dataLoader:
    print(i.shape)
    print(model(i))
    break
# for epoch in range(N_EPOCH):
    
#     total_loss = 0

#     for i, pixel in enumerate(dataLoader):
        
#         (encoded, decoded) = autoencoder(pixel)

#         loss = loss_fn(decoded, pixel)
#         total_loss += loss * 100

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     if(total_loss < min_total_loss):
#         torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': autoencoder.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': total_loss,
#                 }, PATH)
#         min_total_loss = total_loss
    

#     if(epoch % 10 == 0 and epoch > 0):
#         print(epoch, total_loss / len(dataLoader))
