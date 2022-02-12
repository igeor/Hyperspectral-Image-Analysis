import torch
import torch.nn as nn
import h5py
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image
from models import Encoder, Decoder
import atexit

from __init__ import *


def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))
    
parser = argparse.ArgumentParser()
parser.add_argument("-fout", "--outfilename", default='out')
parser.add_argument("-bs", "--batchsize", default=32)
parser.add_argument("-lr", "--lrate", default=0.0002)
parser.add_argument("-ep", "--epochs", default=100)
parser.add_argument("-tr", "--ntrain", default=10)
args = parser.parse_args()

device = "cpu"
if(torch.cuda.is_available()):
    print('---device:cuda---')
    device = "cuda"
else:
    print('---device:cpu---')

f = h5py.File(parentdir + "\\" + "data\panagia.h5", 'r')
dset = f['dataset']
energies = f['energies']
spectra_panagia = np.array(dset)
spectra_panagia = spectra_panagia[:, 60:900]
energies = np.array(energies)
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
inTrainImageMean = torch.mean(inTrainImage, dim=1)
inTrainImageStd = torch.std(inTrainImage, dim=1)
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
inTestImageMean = torch.mean(inTestImage, dim=1)
inTestImageStd = torch.std(inTestImage, dim=1)
# print(inTrainImage.shape)
# print(inTestImage.shape)

outTestImage = TF.to_tensor(Image.open(parentdir + "\\" + "data\jesus.png"))[:3, :, :]
outTestImage_flat = TF.to_tensor(Image.open(parentdir + "\\" + "data\jesus.png"))[:3, :, :]
outTestImage = torch.reshape(outTestImage_flat, (3, 31 * 46))
# print(outTestImage.shape)

# hyperparameters
int(args.batchsize)
DISPLAY_STEP = 200
float(args.lrate) #= 0.00002
N_PIXEL = 21 * 33
TEST_PIXEL = 31 * 46
PATH = "model.pt"

avg_psnr_metric = 0
total_max_psnr_metric = 0
for i_train in range(int(args.ntrain)):
    
    encoder = Encoder(840,3).to(device)
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=float(args.lrate))
    decoder = Decoder(3,840).to(device)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=float(args.lrate))

    L1Loss = nn.L1Loss()
    L2Loss = nn.MSELoss()

    ##### TRAINING #####
    max_psnr_metric = -1
    for epoch in range(int(args.epochs)):
        
        total_loss = 0
        total_l2Loss_dec,total_l2Loss_enc = 0, 0
        
        for i in range(0, N_PIXEL, int(args.batchsize)):

            pixel = inTrainImage[:, i:i+int(args.batchsize)].to(device) / max(energies)
            pixel_real = outTrainImage[:, i:i+int(args.batchsize)].to(device)

            pixel = torch.swapaxes(pixel, 0, 1)
            pixel_real = torch.swapaxes(pixel_real, 0, 1)

            ### Update Decoder ###
            dec_optimizer.zero_grad()
            with torch.no_grad():
                x1,x2,x3,x4,encoded  = encoder(pixel)

            decoded = decoder(x1,x2,x3,x4,encoded.detach())
            dec_loss = L1Loss(decoded, pixel) 

            ### Update Encoder ###
            enc_optimizer.zero_grad()
            x1,x2,x3,x4,encoded  = encoder(pixel)
            decoded = decoder(x1,x2,x3,x4,encoded)
            enc_loss = L1Loss(decoded, pixel) 
            enc_loss = L2Loss(encoded, pixel_real) * 100 # * (1 - epoch/int(args.epochs))* 100
                
            enc_loss.backward()
            enc_optimizer.step()
            dec_loss.backward()
            dec_optimizer.step()

            

        ##### VALIDATING #####
        out_img = torch.zeros(3, 31, 46)
        for i in range(TEST_PIXEL):
            pixel = inTestImage[:, i].to(device) / max(energies)
            r = i // 46; c = i % 46

            with torch.no_grad():
                x1,x2,x3,x4,encoded = encoder(pixel, train=False)
            
            out_img[:, r, c] = encoded
            
        ##### SHOW CURRENT RESULTS #####    
        psnr_metric = PSNR(out_img, outTestImage_flat)
        if(psnr_metric.item() > max_psnr_metric):
            max_psnr_metric = psnr_metric.item()
            best_epoch = epoch
            best_out_img = out_img
            save_image(out_img, './results/spec2spec/best_'+
                args.outfilename+'-'
                'bs_'+str(int(args.batchsize))+'-'+
                'lr_'+str(float(args.lrate))+'-'+
                '.png')
        
        if(psnr_metric.item() > total_max_psnr_metric):
            total_max_psnr_metric = psnr_metric.item()
            total_best_out_img = out_img

        print(" |itrain "+str(i_train)+ 
              " |epoch "+str(epoch)+
              " |loss "+str(round(dec_loss.item(),6))+
              " |psnr "+str(round(max_psnr_metric,2)), end="\r")
    
    print()
    avg_psnr_metric += max_psnr_metric / int(args.ntrain)
    
print(" === AVERAGE_PSNR:"+str(round(avg_psnr_metric,2))+" ===")
save_image(total_best_out_img, './results/spec2spec/best_'+
                args.outfilename+'-'
                'psnr_'+str(round(avg_psnr_metric, 2))+'-'+
                'bs_'+str(int(args.batchsize))+'-'+
                'lr_'+str(float(args.lrate))+'-'+
                '.png')