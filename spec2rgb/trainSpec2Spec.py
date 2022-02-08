import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import h5py
from PIL import Image

from __init__ import *

def h5toNumpy(filepath):
    f = h5py.File(filepath, 'r')
    image = np.array(f['dataset'])
    energies = np.array(f['energies'])
    f.close()
    return image



def saveImage(image, name):
    if(type(image) == type(torch.zeros((1,2)))):
        plt.imshow(image, interpolation='nearest')
    else:
        plt.imshow(image.reshape(image.shape[1],image.shape[2], image.shape[0]), interpolation='nearest')
    plt.savefig(name)


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])



class CustomDataset(Dataset):
        def __init__(self):
            self.train = True 

            self.inTrainImagePath = parentdir + "\\" + "data\jesus.h5"
            self.outTrainImagePath = parentdir + "\\" + "data\jesus.png"
            self.inTestImagePath = parentdir + "\\" + "data\panagia.h5"
            self.outTestImagePath = parentdir + "\\" + "data\panagia.png"
            self.h5TrainShape = (31, 46, 2048)
            self.h5TestShape = (21, 33, 2048)

            self.inTrainImage = h5toNumpy(self.inTrainImagePath).astype(np.float32)
            self.inTrainImage = torch.from_numpy(self.inTrainImage)
            self.inTrainImage = torch.reshape(self.inTrainImage, (31,46,2048))
            self.inTrainImage = torch.swapaxes(self.inTrainImage,0,2);	self.inTrainImage = np.swapaxes(self.inTrainImage,1,2)
            self.inTrainImage = torch.rot90(self.inTrainImage, 2, (2,1))
            self.pltInTrainImage = torch.sum(self.inTrainImage, axis=0)
            self.inTrainImage = self.inTrainImage.reshape(2048, 31 * 46)  

            self.inTestImage = h5toNumpy(self.inTestImagePath).astype(np.float32)
            self.inTestImage = torch.from_numpy(self.inTestImage)
            self.inTestImage = torch.reshape(self.inTestImage, (21,33,2048))
            self.inTestImage = torch.swapaxes(self.inTestImage,0,2);	self.inTestImage = np.swapaxes(self.inTestImage,1,2)
            self.inTestImage = torch.rot90(self.inTestImage, 2, (2,1))
            self.pltInTestImage = torch.sum(self.inTestImage, axis=0)
            self.inTestImage = self.inTestImage.reshape(2048, 21 * 33)

            self.outTestImage = torch.from_numpy(np.array(Image.open(self.outTestImagePath).convert('RGB')).flatten().reshape(3, 21 * 33))
            self.outTrainImage = torch.from_numpy(np.array(Image.open(self.outTrainImagePath).convert('RGB')).flatten().reshape(3, 31 * 46))
            # print(self.inTrainImage.shape)
            # print(self.inTestImage.shape)
            # print(self.outTestImage.shape)
            # print(self.outTrainImage.shape)
        
        def setTrain(self, train):
            self.train = train

        def __len__(self):
            if(not self.train):
                return self.inTestImage.shape[-1]
            return self.inTrainImage.shape[-1]
        
        def __getitem__(self, index):
            if(not self.train):
                return self.inTestImage[:,index], self.outTestImage[:, index]
            return self.inTrainImage[:, index], self.outTrainImage[:, index]


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512), 
            nn.ReLU(True),
            nn.Linear(512, 2048)
        )

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_enc, x_dec


model = autoencoder()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.02, weight_decay=1e-5)

for epoch in range(num_epochs):
    for spec, rgb in dataloader:
        
        # ===================forward=====================
        enc, dec = model(spec)
        loss = criterion(dec, spec)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss))
        