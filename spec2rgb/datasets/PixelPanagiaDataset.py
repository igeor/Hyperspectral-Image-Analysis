import torch
from PIL import Image
import torchvision.transforms.functional as TF
import h5py

from __init__ import *



class PixelPanagiaDataset(Dataset):

    def __init__(self,
        inImagePath,
        outImagePath,
        inSpectraPath = None,
        inTrainImaegShape=None,
        inImageShape=None):
        
        self.inImage = torch.tensor(np.load(inImagePath))

        self.outImage = Image.open(outImagePath).convert('RGB')
        self.outImage = TF.to_tensor(self.outImage)
        self.outImage2d = self.outImage
        self.outImage = torch.flatten(self.outImage, start_dim=1)

        
        f = h5py.File(inSpectraPath, 'r')
        self.energies = np.array(f['energies'])
        f.close()
        
        
    def setTrain(self, train):
        self.train = train
    
    def __getitem__(self, index):
        return (self.inImage[index,:] / max(self.energies), self.outImage[:,index])

    def __len__(self):
        return self.inImage.shape[0]

