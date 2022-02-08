
import torch
import torchvision
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from __init__ import *

device = "cpu"

f = h5py.File(parentdir + "\\" + "data\panagia.h5",  'r')
dset = f['dataset']
energies = f['energies']
spectra_panagia = np.array(dset)
spectra_panagia = spectra_panagia[:,60:900]
energies = np.array(energies)
f.close()


import torchvision.transforms.functional as TF

inTrainImage = spectra_panagia.astype(np.float32)
inTrainImage = torch.from_numpy(inTrainImage)
inTrainImage = torch.reshape(inTrainImage, (21,33,840))
inTrainImage = torch.swapaxes(inTrainImage,0,2);	inTrainImage = np.swapaxes(inTrainImage,1,2)
inTrainImage = torch.rot90(inTrainImage, 2, (2,1))
inTrainImage = torch.reshape(inTrainImage, (840, 21 * 33))
print(inTrainImage.shape)

outTrainImage = TF.to_tensor(Image.open(parentdir + "\\" + "data\panagia.png"))[:3,:,:]
outTrainImage = torch.reshape(outTrainImage, (3, 21 * 33))
print(outTrainImage.shape)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(840, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 840)
        )

    def forward(self, pixel):
        encoded = self.encoder(pixel)
        decoded = self.decoder(encoded)
        return encoded, decoded



# hyperparameters
N_EPOCH = 100
BATCH_SIZE = 1
DISPLAY_STEP = 200
learning_rate = 0.0002
N_PIXEL = 693
PATH = "model.pt"


autoencoder = AutoEncoder().to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
loss_fn = nn.L1Loss()


out_img = torch.zeros(3,21,33)
min_total_loss = float('inf')
total_loss = 0

for epoch in range(N_EPOCH):
    for i in range(N_PIXEL):
        pixel = inTrainImage[:,i]
        pixel_real = outTrainImage[:,i]
        (encoded, decoded) = autoencoder(pixel)

        total_loss += loss_fn(decoded, pixel) 

        if(i % DISPLAY_STEP == 0 and i > 0):
            print("epoch ",str(epoch)+": "+str(total_loss.item()))#, end="\r")
            #save_image(out_img, "results/" + str(epoch)+"_"+str(i)+"_"+'out.png')

                
        if(i % BATCH_SIZE == 0 and i > 0):
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_loss = 0
        
       

        r = i // 33 ; c = i % 33
        #print(r,c)
        out_img[:, r, c] = encoded
        save_image(out_img, 'out.png')
        
        
