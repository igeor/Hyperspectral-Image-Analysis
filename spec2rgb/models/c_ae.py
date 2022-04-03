import torch
import torch.nn as nn


class cEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, cancel_out_layer):
        super(cEncoder, self).__init__()
        
        self.cancel_out_layer = cancel_out_layer

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
        #print(x_in.device)
        x1 = self.cancel_out_layer(x_in)

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