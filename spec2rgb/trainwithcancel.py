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
    mse = torch.mean((img1*255.0 - img2*255.0) ** 2)
    
    return 20 * torch.log10(255.0 / torch.sqrt(mse))
    
parser = argparse.ArgumentParser()
parser.add_argument("-fout", "--outfilename", default='out')
parser.add_argument("-bs", "--batchsize", default=32)
parser.add_argument("-lr", "--lrate", default=0.0002)
parser.add_argument("-ep", "--epochs", default=100)
parser.add_argument("-tr", "--ntrain", default=10)
parser.add_argument("-ae", "--autoencoder", default=False)

args = parser.parse_args()

device = "cpu"
if(torch.cuda.is_available()):
    device = "cuda"; print('---device:cuda---')


        
trainSet = PixelPanagiaDataset(
    inImagePath = parentdir + "\\" + "data\\panagia.npy",
    outImagePath = parentdir + "\\" + "data\panagia.png",
    inSpectraPath =  parentdir + "\\" + "data\panagia.h5"
)
testSet = PixelPanagiaDataset(
    inImagePath = parentdir + "\\" + "data\\jesus.npy",
    outImagePath = parentdir + "\\" + "data\jesus.png",
    inSpectraPath =  parentdir + "\\" + "data\jesus.h5"
)

trainLoader = DataLoader(dataset=trainSet, batch_size=32, shuffle=True)
testLoader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)














def h5toNumpy(filepath):
    f = h5py.File(filepath, 'r')
    image = np.array(f['dataset'])
    energies = np.array(f['energies'])
    f.close()
    return image

toTensor = torchvision.transforms.ToTensor()

inTrainImage = h5toNumpy(parentdir + "\\" + "data\\panagia.h5").astype(np.float32)
inTrainImage = torch.from_numpy(inTrainImage)
inTrainImage = torch.reshape(inTrainImage, [21, 33, 2048])
inTrainImage = torch.swapaxes(inTrainImage,0,2);	inTrainImage = np.swapaxes(inTrainImage,1,2)
inTrainImage = torch.rot90(inTrainImage, 2, (2,1))

outTrainImage = toTensor(Image.open(parentdir + "\\" + "data\panagia.png").convert('RGB'))

inTestImage = h5toNumpy(parentdir + "\\" + "data\\jesus.h5").astype(np.float32)
inTestImage = torch.from_numpy(inTestImage)
inTestImage = torch.reshape(inTestImage, [31, 46, 2048])
inTestImage = torch.swapaxes(inTestImage,0,2);	inTestImage = np.swapaxes(inTestImage,1,2)
inTestImage = torch.rot90(inTestImage, 2, (2,1))

outTestImage = toTensor(Image.open(parentdir + "\\" + "data\jesus.png").convert('RGB'))

class CustomDataset(Dataset):
    
    def __init__(self,
        inTrainImage,
        outTrainImage,
        inTestImage,
        outTestImage):
        
        self.inTrainImage = inTrainImage
        self.outTrainImage = outTrainImage
        self.inTestImage = inTestImage
        self.outTestImage = outTestImage
        self.outTestImage2D = outTestImage
       
        self.inTrainImage = torch.flatten(self.inTrainImage, start_dim=1, end_dim=-1)
        self.outTrainImage = torch.flatten(self.outTrainImage, start_dim=1, end_dim=-1)
        self.inTestImage = torch.flatten(self.inTestImage, start_dim=1, end_dim=-1)
        self.outTestImage = torch.flatten(self.outTestImage, start_dim=1, end_dim=-1)
        
        print(
            self.inTrainImage.shape,
            self.outTrainImage.shape,
            self.inTestImage.shape,
            self.outTestImage.shape
        )
        
    def setTrain(self, train):
        self.train = train
    
    def __getitem__(self, index):
        if(not self.train):
            return (self.inTrainImage[60:900,index], self.outTrainImage[:,index])
        return (self.inTestImage[60:900,index], self.outTestImage[:,index])
    
    def __len__(self):
        return self.inTrainImage.shape[-1]
      

pDataset = CustomDataset(inTrainImage,outTrainImage,inTestImage,outTestImage)
dataloader = DataLoader(dataset=pDataset, batch_size=32, shuffle=True)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

avg_psnr_metric = 0
total_max_psnr_metric = 0
for i_train in range(int(args.ntrain)):
    
    encoder = cEncoder(840,3, CancelOut(840).to(device)).to(device)
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=float(args.lrate))
    decoder = Decoder(3,840).to(device)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=float(args.lrate))

    loss_1 = nn.SmoothL1Loss()
    loss_2 = nn.MSELoss()

    ##### TRAINING #####
    max_psnr_metric = -1
    for epoch in range(int(args.epochs)):
        
        pDataset.setTrain(True)
        for pixel , pixel_real in dataloader:

            pixel, pixel_real = pixel.to(device), pixel_real.to(device)

            ### Decoder Loss Estimation ###
            dec_loss = 0
            for step in range(1):
                if(bool(args.autoencoder)):
                    dec_optimizer.zero_grad()
                    with torch.no_grad():
                        x1,x2,x3,x4,x5,encoded  = encoder(pixel)
                    decoded = decoder(x1,x2,x3,x4,x5,encoded.detach())
                    dec_loss += loss_1(decoded, pixel) 

            ### Encoder Loss Estimation ###
            enc_optimizer.zero_grad()
            x1,x2,x3,x4,x5,encoded  = encoder(pixel)
            enc_loss = loss_2(encoded, pixel_real) #* 100 #* 100  # * (1 - epoch/int(args.epochs))* 100
            
            if(bool(args.autoencoder)): 
                # add decoder's loss to encoder 
                # in order to act as autoencoder
                with torch.no_grad():
                    decoded = decoder(x1,x2,x3,x4,x5,encoded)
                enc_loss += loss_1(decoded, pixel) 
            
            ### Update Models Weights ###
            enc_loss.backward()
            enc_optimizer.step()
            if(bool(args.autoencoder)):
                dec_loss.backward()
                dec_optimizer.step()

            

        ##### VALIDATING #####
        out_img = torch.zeros(3, 31, 46)
        pDataset.setTrain(False)
        for i, (pixel, pixel_real) in enumerate(dataloader):
            
            pixel, pixel_real = pixel.to(device), pixel_real.to(device)
            r = i // 46; c = i % 46

            with torch.no_grad():
                x1,x2,x3,x4,x5,encoded = encoder(pixel, train=False)
            
            for enc_pixel in encoded:
                out_img[:, r, c] = enc_pixel

            
        ##### SHOW CURRENT RESULTS #####    
        psnr_metric = PSNR(out_img, testSet.outImage2d)
        if(psnr_metric.item() > max_psnr_metric):
            max_psnr_metric = psnr_metric.item()
            best_epoch = epoch
            best_out_img = out_img
            save_image(out_img, './results/spec2spec/curr_'+
                args.outfilename+'-'
                'bs_'+str(int(args.batchsize))+'-'+
                'lr_'+str(float(args.lrate))+'-'+
                '.png')
        
        if(psnr_metric.item() > total_max_psnr_metric):
            total_max_psnr_metric = psnr_metric.item()
            total_best_out_img = out_img

        print(" | i_train "+str(i_train)+ 
              " | epoch "+str(epoch)+
              " | enc_loss "+str(round(enc_loss.item(),6))+
              " | psnr "+str(round(max_psnr_metric,2)), end="\r")
    
    print()
    avg_psnr_metric += max_psnr_metric / int(args.ntrain)
    
print(" === AVERAGE_PSNR:"+str(round(avg_psnr_metric,2))+" ===")
save_image(total_best_out_img, './results/spec2spec/best_'+
                args.outfilename+'-'
                'psnr_'+str(round(avg_psnr_metric, 2))+'-'+
                'bs_'+str(int(args.batchsize))+'-'+
                'lr_'+str(float(args.lrate))+'-'+
                '.png')