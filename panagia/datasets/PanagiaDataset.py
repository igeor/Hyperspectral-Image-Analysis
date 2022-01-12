import torch
from torch.utils.data.dataset import Dataset
import numpy as np 

from .utils.h5toNumpy import h5toNumpy

class PanagiaDataset(Dataset):
    
    def __init__(self, path, block_size_r = 6, block_size_c = 9, step=1):
        
        self.path = path 
        self.block_size_r = block_size_r
        self.block_size_c = block_size_c
        self.step = step 
        
        npImage = h5toNumpy([path])[0][0]
        npEnergies = h5toNumpy([path])[0][1]

        self.x_in = npImage.reshape((21,33,2048))
        self.x_in = torch.Tensor(self.x_in.astype(np.float32)).to("cuda")

        self.blocks = list()
        
        # Crop out the block and append to blocks
        for r in range(0,self.x_in.shape[0] - block_size_r, step):
            for c in range(0,self.x_in.shape[1] - block_size_c, step):
                self.blocks.append(self.x_in[r:r+block_size_r,c:c+block_size_c, :])
        
    def __getitem__(self, index):
        return self.blocks[index]

    def __len__(self):
        return len(self.blocks)
