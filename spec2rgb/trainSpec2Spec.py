import torch
import torch.nn as nn
import h5py
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image
from models import Encoder, Decoder

from __init__ import *

device = "cpu"
if(torch.cuda.is_available()):
    device = "cuda"
    
f = h5py.File(parentdir + "\\" + "data\panagia.h5", 'r')
dset = f['dataset']
energies = f['energies']
spectra_panagia = np.array(dset)
spectra_panagia = spectra_panagia[:, 60:900]
energies = np.array(energies)
print(energies,energies.shape);input()
f.close()

f2 = h5py.File(parentdir + "\\" + "data\jesus.h5", 'r')
dset2 = f2['dataset']
energies2 = f2['energies']
spectra_panagia2 = np.array(dset2)
spectra_panagia2 = spectra_panagia2[:, 60:900]
energies2 = np.array(energies2)
f2.close()

inTrainImage = spectra_panagia.astype(np.float32)
inTrainImage = torch.from_numpy(inTrainImage)
inTrainImage = torch.reshape(inTrainImage, (21, 33, 840))
inTrainImage = torch.swapaxes(inTrainImage, 0, 2)
inTrainImage = np.swapaxes(inTrainImage, 1, 2)
inTrainImage = torch.rot90(inTrainImage, 2, (2, 1))
inTrainImage = torch.reshape(inTrainImage, (840, 21 * 33))
# print(inTrainImage.shape)

outTrainImage = TF.to_tensor(Image.open(parentdir + "\\" + "data\panagia.png"))[:3, :, :]
outTrainImage = torch.reshape(outTrainImage, (3, 21 * 33))
# print(outTrainImage.shape)

inTestImage = spectra_panagia2.astype(np.float32)
inTestImage = torch.from_numpy(inTestImage)
inTestImage = torch.reshape(inTestImage, (31, 46, 840))
inTestImage = torch.swapaxes(inTestImage, 0, 2)
inTestImage = np.swapaxes(inTestImage, 1, 2)
inTestImage = torch.rot90(inTestImage, 2, (2, 1))
inTestImage = torch.reshape(inTestImage, (840, 31 * 46))
# print(inTestImage.shape)

outTestImage = TF.to_tensor(Image.open(parentdir + "\\" + "data\jesus.png"))[:3, :, :]
outTestImage = torch.reshape(outTestImage, (3, 31 * 46))
# print(outTestImage.shape)

# hyperparameters
N_EPOCH = 300
BATCH_SIZE = 33
DISPLAY_STEP = 200
learning_rate = 0.00002
N_PIXEL = 21 * 33
TEST_PIXEL = 31 * 46
PATH = "model.pt"

encoder = Encoder(840,3).to(device)
enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder = Decoder(3,840).to(device)
dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

L1Loss = nn.L1Loss()
L2Loss = nn.MSELoss()

for epoch in range(N_EPOCH):
    
    ##### TRAINING #####
    total_loss = 0
    total_l2Loss_dec,total_l2Loss_enc = 0, 0
    out_img1 = torch.zeros(3, 21, 33)
    for i in range(0, N_PIXEL, BATCH_SIZE):

        pixel = inTrainImage[:, i:i+BATCH_SIZE].to(device) / max(energies)
        pixel_real = outTrainImage[:, i:i+BATCH_SIZE].to(device)

        pixel = torch.swapaxes(pixel, 0, 1)
        pixel_real = torch.swapaxes(pixel_real, 0, 1)

        ### Update Decoder ###
        dec_optimizer.zero_grad()
        with torch.no_grad():
            x1,x2,x3,x4,encoded  = encoder(pixel)

        decoded = decoder(x1,x2,x3,x4,encoded.detach())
        dec_loss = L1Loss(decoded, pixel)
        #dec_loss += L2Loss(encoded.detach(), pixel_real) / (N_PIXEL // BATCH_SIZE)

        ### Update Encoder ###
        enc_optimizer.zero_grad()
        x1,x2,x3,x4,encoded  = encoder(pixel)
        decoded = decoder(x1,x2,x3,x4,encoded)
        enc_loss = L1Loss(decoded, pixel) 
        enc_loss += L2Loss(encoded, pixel_real) * (1 - epoch/N_EPOCH) * 100

        
            
        enc_loss.backward()
        enc_optimizer.step()
        dec_loss.backward()
        dec_optimizer.step()

        for x, j in enumerate(range(i, i+BATCH_SIZE)):
            r = j // 33;c = j % 33
            out_img1[:, r, c] = encoded[x]

        
    ##### TESTING #####
    TEST_BATCH_SIZE = 46
    out_img2 = torch.zeros(3, 31, 46)
    for i in range(0, TEST_PIXEL, TEST_BATCH_SIZE):

        pixel = inTestImage[:, i:i+TEST_BATCH_SIZE].to(device)  / max(energies)
        pixel_real = outTrainImage[:, i:i+TEST_BATCH_SIZE].to(device)

        pixel = torch.swapaxes(pixel, 0, 1)
        pixel_real = torch.swapaxes(pixel_real, 0, 1)

        with torch.no_grad():
            x1,x2,x3,x4,encoded = encoder(pixel)

        for x, j in enumerate(range(i, i+TEST_BATCH_SIZE)):
            r = j // 46
            c = j % 46
            out_img2[:, r, c] = encoded[x]

    ##### VIEW RESULTS #####
    print("epoch ", str(epoch)+": "+str(dec_loss.item()), end="\r")
    save_image(out_img1, 'out1.png')
    save_image(out_img2, 'out2.png')