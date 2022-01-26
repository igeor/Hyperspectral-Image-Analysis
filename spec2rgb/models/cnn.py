from torch import nn

import torch.nn as nn
import torch

class ConvNN(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=2048, out_channels=3):
        """Initializes U-Net."""

        super(ConvNN, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
        	nn.Conv2d(in_channels, 1024, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.Dropout(0.2),
        	nn.ReLU(inplace=True)
            # nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
        	# nn.ReLU(inplace=True)
        )
        
        self._block2 = nn.Sequential(
        	nn.Conv2d(1024, 512, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.Dropout(0.2),
        	nn.ReLU(inplace=True)
            # nn.Conv2d(512, 512, 3, stride=1, padding=1),
        	# nn.ReLU(inplace=True)
        )
  
        self._block3 = nn.Sequential(
        	nn.Conv2d(512, 256, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.Dropout(0.2),
        	nn.ReLU(inplace=True)
            # nn.Conv2d(256, 256, 3, stride=1, padding=1),
        	# nn.ReLU(inplace=True)
        )
        
        self._block4 = nn.Sequential(
        	nn.Conv2d(256, 128, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.Dropout(0.2),
        	nn.ReLU(inplace=True)
            # nn.Conv2d(128, 128, 3, stride=1, padding=1),
        	# nn.ReLU(inplace=True)
        )
        
        self._block5 = nn.Sequential(
        	nn.Conv2d(128, 64, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(64, 64, 3, padding=1),
            nn.Dropout(0.2),
        	nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True)
        )
            
        self._block6 = nn.Sequential(
        	nn.Conv2d(64, 32, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(32, 32, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True)
        )
        
        self._block7 = nn.Sequential(
        	nn.Conv2d(3, 32, 3, stride=2, padding=1),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(32, 32, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True)
        )
        
        
        self._block8 = nn.Sequential(
        	nn.Conv2d(32, 64, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(64, 64, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True)
        )
        
        self._block9 = nn.Sequential(
        	nn.Conv2d(64, 32, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(32, 32, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True)
        )
        
        self._block10 = nn.Sequential(
        	nn.ConvTranspose2d(64, out_channels, 3, stride=2, padding=1),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True)
        )
        
        self._block11 = nn.Sequential(
        	nn.Conv2d(out_channels * 2, out_channels, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        	nn.ReLU(inplace=True)
        )
        
        
        # Initialize weights
        #self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        # Encoder
        x1 = self._block1(x)
        #print(x1.shape)
        x2 = self._block2(x1)
        #print(x2.shape)
        x3 = self._block3(x2)
        #print(x3.shape)
        x4 = self._block4(x3)
        #print(x4.shape)
        x5 = self._block5(x4)
        #print(x5.shape)
        x6 = self._block6(x5)
        #print(x6.shape)
        x7 = self._block7(x6)
        #print(x7.shape)
        x8 = self._block8(x7)
        #print(x8.shape)
        x9 = self._block9(x8)
        #print(x9.shape)
        x10 = self._block10(torch.cat([x9, x7], dim=1))
        #print(x10.shape)
        x11 = self._block11(torch.cat([x10, x6], dim=1))
        #print(x11.shape)
        return x11