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
            nn.Linear(in_channels//16, in_channels//32),
            nn.ReLU()
            #nn.BatchNorm1d(in_channels//16)
        )
        self.l6 =  nn.Sequential(
            nn.Linear(in_channels//32, out_channels),
        )
        self.dropout = nn.Dropout(0.0)
        
    def forward(self, x_in, train=True):
        x1 = self.l1(x_in)
        #print(x1.shape)
        x2 = self.l2(x1); 
        if(train):
            x2 == self.dropout(x2)
        #print(x2.shape)
        x3 = self.l3(x2); 
        if(train):
            x3 == self.dropout(x3)
        #print(x3.shape)
        x4 = self.l4(x3)
        if(train):
            x4 == self.dropout(x4)
        x5 = self.l5(x4)
        #print(x4.shape)
        if(train):
            x5 == self.dropout(x5)
        #print(x4.shape)
        x_enc = self.l6(x5)
        return x1,x2,x3,x4,x5,x_enc 


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.l1 =  nn.Sequential(
            nn.Linear(in_channels, out_channels//32),
            nn.ReLU()
            #nn.BatchNorm2d(out_channels//16)
        )
        self.l2 =  nn.Sequential(
            nn.Linear(out_channels//32, out_channels//16),
            nn.ReLU()
            #nn.BatchNorm2d(out_channels//16)
        )
        self.l3 =  nn.Sequential(
            nn.Linear(out_channels//16, out_channels//8),
            nn.ReLU()
            #nn.BatchNorm1d(out_channels//8)
        )
        self.l4 =  nn.Sequential(
            nn.Linear(out_channels//8, out_channels//4),
            nn.ReLU()
            #nn.BatchNorm1d(out_channels//4)
        )
        self.l5 =  nn.Sequential(
            nn.Linear(out_channels//4, out_channels//2),
            nn.ReLU()
            #nn.BatchNorm1d(out_channels//2)
        )
        self.l6 =  nn.Sequential(
            nn.Linear(out_channels//2, out_channels),
        )

        self.dropout = nn.Dropout(0.0)
        
    def forward(self, x1_e,x2_e,x3_e,x4_e,x5_e,x_in, train=True):
        x1 = self.l1(x_in)
        #print(x1.shape, x5_e.shape)
        x2 = self.l2(x1 + x5_e); 
        if(train):
            x2 == self.dropout(x2)
        #print(x2.shape)    
        x3 = self.l3(x2 + x4_e); 
        if(train):
            x3 == self.dropout(x3)
        #print(x3.shape)
        x4 = self.l4(x3 + x3_e); 
        if(train):
            x4 == self.dropout(x4)
        #print(x4.shape)
        x5 = self.l5(x4 + x2_e); 
        if(train):
            x5 == self.dropout(x5)
        #print(x5.shape)
        x_dec = self.l6(x5)
        return x_dec 