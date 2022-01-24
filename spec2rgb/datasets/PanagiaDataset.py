from locale import normalize
from __init__ import *
from .utils import h5toNumpy, npRotate

from PIL import Image
import h5py
import numpy as np
import os
from torchvision import transforms 
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def h5toNumpy(filepath):
    f = h5py.File(filepath, 'r')
    image = np.array(f['dataset'])
    energies = np.array(f['energies'])
    f.close()
    return image



class Patch: 
    
    def __init__(self, arr, r_index , c_index):
        self.arr = arr 
        self.r_index = r_index
        self.c_index = c_index
        self.w = self.arr.shape[2]
        self.h = self.arr.shape[1]
        self.flat = torch.sum(arr, axis=0)
        self.dict = {
            "img":self.arr,
            "r":self.r_index,
            "c":self.c_index
        }
        self.batch = (self.arr, self.r_index, self.c_index)
class PanagiaDataset(Dataset):
    
    def __init__(
        self, 
        inTrainImagePath, 
        outTrainImagePath,
        inTestImagePath, 
        outTestImagePath,
        h5TrainShape, 
        h5TestShape,
        isTrain = True):
        
        self.toTensor = transforms.ToTensor()
        self.isTrain = isTrain 
        
        self.inTrainImage = h5toNumpy(inTrainImagePath).astype(np.float32)
        self.inTrainImage = torch.from_numpy(self.inTrainImage)
        self.inTrainImage = torch.reshape(self.inTrainImage, h5TrainShape)
        self.inTrainImage = torch.swapaxes(self.inTrainImage,0,2);	self.inTrainImage = np.swapaxes(self.inTrainImage,1,2)
        self.inTrainImage = torch.rot90(self.inTrainImage, 2, (2,1))
        self.pltInTrainImage = torch.sum(self.inTrainImage, axis=0)
        # self.showImage(self.pltInTrainImage)
        self.saveImage(self.pltInTrainImage, 'inTrainImage.png')

        self.outTrainImage = self.toTensor(Image.open(outTrainImagePath))
        save_image(self.outTrainImage , 'outTrainImage.png')
        
        self.inTestImage = h5toNumpy(inTestImagePath).astype(np.float32)
        self.inTestImage = torch.from_numpy(self.inTestImage)
        self.inTestImage = torch.reshape(self.inTestImage, h5TestShape)
        self.inTestImage = torch.swapaxes(self.inTestImage,0,2);	self.inTestImage = np.swapaxes(self.inTestImage,1,2)
        self.inTestImage = torch.rot90(self.inTestImage, 2, (2,1))
        self.pltInTestImage = torch.sum(self.inTestImage, axis=0)
        #self.showImage(self.pltInTestImage)
        self.saveImage(self.pltInTestImage, 'inTestImage.png')
        
        self.outTestImage = self.toTensor(Image.open(outTestImagePath).convert('RGB'))
        save_image(self.outTestImage , 'outTestImage.png')
           
        print(self.inTrainImage.shape, type(self.inTrainImage))
        print(self.outTrainImage.shape, type(self.outTrainImage))
        print(self.inTestImage.shape, type(self.inTestImage))
        print(self.outTestImage.shape, type(self.outTestImage))
        
        self.wTrain = h5TrainShape[1]
        self.hTrain = h5TrainShape[0]
        self.wTest = h5TestShape[1]
        self.hTest = h5TestShape[0]

        self.outTrainRecon = torch.zeros((3,self.hTrain,self.wTrain))
        self.outTestRecon = torch.zeros((1,3,self.hTest,self.wTest))

        self.inTrainPatches = list()
        self.outTrainPatches = list()
        self.inTestPatches = list()
        self.outTestPatches = list()

        r = 15; c = 19; step = 1
        self.r = 15 ; self.c = 19
        stepR = self.hTest // r
        stepC = self.wTest // c

        for i in range(0, self.wTrain - c, step):
            for j in range(0,  self.hTrain - r, step):

                inTrainPatch = self.inTrainImage[:, j:j+r, i:i+c] 
                inTrainPatch = Patch(inTrainPatch, r_index=j, c_index=i)
                self.inTrainPatches.append(inTrainPatch)
                
                outTrainPatch = self.outTrainImage[:, j:j+r, i:i+c] 
                outTrainPatch = Patch(outTrainPatch, r_index=j, c_index=i)
                self.outTrainPatches.append(outTrainPatch)

        for i in range(0, self.wTest - c, stepR):
            for j in range(0,  self.hTest - r, stepC):

                inTestPatch = self.inTestImage[:, j:j+r, i:i+c] 
                inTestPatch = Patch(inTestPatch, r_index=j, c_index=i)
                self.inTestPatches.append(inTestPatch)
                
                outTestPatch = self.outTestImage[:, j:j+r, i:i+c] 
                outTestPatch = Patch(outTestPatch, r_index=j, c_index=i)
                self.outTestPatches.append(outTestPatch)
        


    def setTrain(self, isTrain):
        self.isTrain = isTrain 
        
    def reconstructFromPatches(self, patch, r_index, c_index):
        self.outTestRecon[:,:,r_index:r_index + self.r  , c_index:c_index+self.c] = patch[0]
        

    def __getitem__(self, index):
        if(not self.isTrain):
            return (self.inTestPatches[index].dict, self.outTestPatches[index].dict)
        else:
            return (self.inTrainPatches[index].dict, self.outTrainPatches[index].dict)
    
    
    def __len__(self):
        if(not self.isTrain):
            return len(self.inTestPatches)
        return len(self.inTrainPatches)




    def showImage(self, image):
        plt.imshow(image, interpolation='nearest')
        plt.show()
    
    def showRecImage(self):
        plt.imshow(torch.sum(self.outTestRecon[0], axis=0), interpolation='nearest')
        save_image(self.outTestRecon, 'outtorch.png')
        plt.imshow(torch.sum(self.outTestRecon[0], axis=0), interpolation='nearest')
        plt.savefig('out.png')
        
    def saveImage(self, image, name):
        if(type(image) == type(torch.zeros((1,2)))):
            plt.imshow(image, interpolation='nearest')
        else:
            plt.imshow(image.reshape(image.shape[1],image.shape[2], image.shape[0]), interpolation='nearest')
        plt.savefig(name)
