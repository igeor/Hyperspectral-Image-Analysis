import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.l1 =  nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.ReLU()
            #nn.BatchNorm1d(in_channels//2)
        )
        self.l2 =  nn.Sequential(
            nn.Linear(in_channels//2, in_channels//4),
            nn.ReLU()
            #nn.BatchNorm1d(in_channels//4)
        )
        self.l3 =  nn.Sequential(
            nn.Linear(in_channels//4, in_channels//8),
            nn.ReLU()
            #nn.BatchNorm1d(in_channels//8)
        )
        self.l4 =  nn.Sequential(
            nn.Linear(in_channels//8, in_channels//16),
            nn.ReLU()
            #nn.BatchNorm1d(in_channels//16)
        )
        self.l5 =  nn.Sequential(
            nn.Linear(in_channels//16, out_channels),
        )

    def forward(self, x_in, train=True):
        x1 = self.l1(x_in)
        #print(x1.shape)
        x2 = self.l2(x1); 
        if(train):
            x2 == nn.Dropout(0.3)
        #print(x2.shape)
        x3 = self.l3(x2); 
        if(train):
            x2 == nn.Dropout(0.3)
        #print(x3.shape)
        x4 = self.l4(x3)
        #print(x4.shape)
        x_enc = self.l5(x4)
        return x1,x2,x3,x4,x_enc 


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.l1 =  nn.Sequential(
            nn.Linear(in_channels, out_channels//16),
            nn.ReLU()
            #nn.BatchNorm2d(out_channels//16)
        )
        self.l2 =  nn.Sequential(
            nn.Linear(out_channels//16, out_channels//8),
            nn.ReLU()
            #nn.BatchNorm1d(out_channels//8)
        )
        self.l3 =  nn.Sequential(
            nn.Linear(out_channels//8, out_channels//4),
            nn.ReLU()
            #nn.BatchNorm1d(out_channels//4)
        )
        self.l4 =  nn.Sequential(
            nn.Linear(out_channels//4, out_channels//2),
            nn.ReLU()
            #nn.BatchNorm1d(out_channels//2)
        )
        self.l5 =  nn.Sequential(
            nn.Linear(out_channels//2, out_channels),
        )

    def forward(self, x1_e,x2_e,x3_e,x4_e,x_in, train=True):
        x1 = self.l1(x_in)
        #print(x1.shape)
        x2 = self.l2(x1 + x4_e); 
        if(train):
            x2 == nn.Dropout(0.3)
        #print(x2.shape)
        x3 = self.l3(x2 + x3_e); 
        if(train):
            x2 == nn.Dropout(0.3)
        #print(x3.shape)
        x4 = self.l4(x3 + x2_e)
        #print(x4.shape)
        x_dec = self.l5(x4)
        return x_dec 