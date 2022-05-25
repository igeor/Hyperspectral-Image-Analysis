from PIL import Image
from torchvision import transforms 
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import torch 
        
class CustomDataset(Dataset):
    
    def __init__(self, listOfFlattenedImages, numChannels=1948, numElements=8):
        self.numChannels = numChannels
        self.numElements = numElements
        self.images = torch.cat(listOfFlattenedImages, dim=0)


    def __getitem__(self, index):
        return self.images[index, :self.numChannels].float(), self.images[index, self.numChannels:].float()
    
    
    def __len__(self):
        return self.images.shape[0]




