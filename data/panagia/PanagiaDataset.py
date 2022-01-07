import torch
from torch import Dataset

from utils import h5toNumpy

class CustomDataset(Dataset):
    
    def __init__(self, image):
        imgs = h5toNumpy(['panagia.h5'])

        pimg = imgs[0][0]
        pimg = pimg.reshape((21,33,2048))
        
        self.image = torch.Tensor(image.astype(np.float32)).to(device)

    def __getitem__(self, index):
        return self.image[index,:]

    def __len__(self):
        return self.image.shape[0]

    def get_pixel(self, index):
        return self.image[index]