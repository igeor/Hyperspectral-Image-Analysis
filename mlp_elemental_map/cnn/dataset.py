from PIL import Image
from torchvision import transforms 
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset

        
class CustomDataset(Dataset):
    
    def __init__(self, XY_train, XY_test):
        self.isTrain = True 
        self.X_train = XY_train[0]
        self.y_train = XY_train[1]
        self.X_test = XY_test[0]
        self.y_test = XY_test[1]
        
    def setTrain(self,train):
        self.isTrain = train

    def __getitem__(self, index):
        if(self.isTrain):
            return self.X_train[index], self.y_train[index]
        return self.X_test[index], self.y_test[index]
    
    
    def __len__(self):
        if(self.isTrain):
            return self.X_train.shape[0]
        return self.X_test.shape[0]




