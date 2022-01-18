from __init__ import *
from .utils import h5toNumpy, npRotate

class PanagiaDataset(Dataset):
    
    def __init__(
        self, dir="", 
        img_size_r = 31, img_size_c = 46, 
        block_size_r = 6, block_size_c = 9, 
        step=1):
        
        self.dir = dir 
        self.block_size_r = block_size_r
        self.block_size_c = block_size_c
        self.step = step 
        
        npImage = h5toNumpy([dir])[0][0]  
        
        self.img = npImage.reshape((img_size_r,img_size_c,2048))
        self.img = self.img.astype(np.float32)
        
        self.imgBlocks = list()
        
        for nr in [90,180,270]:
            for r in range(0,self.img.shape[0] - block_size_r, step):
                for c in range(0,self.img.shape[1] - block_size_c, step):
                    block = self.img[r:r+block_size_r,c:c+block_size_c, :]
                    block = np.swapaxes(block,0,2);	block = np.swapaxes(block,1,2)
                    block = torchvision.transforms.functional.rotate(torch.tensor(block), nr)
                    self.imgBlocks.append(block)
                        
                        
    def __getitem__(self, index):
        return self.imgBlocks[index]
    
    def __len__(self):
        return len(self.imgBlocks)

