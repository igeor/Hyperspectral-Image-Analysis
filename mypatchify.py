import image_slicer
from PIL import ImageDraw, ImageFont, Image
import numpy as np 
import torch 
import numpy as np
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
from torchvision import transforms 
import h5py
import numpy as np
import os.path 

import sys
import os

'''
@exports:   
    a list of tuples that contain 
    spectra-image and energies 
'''
def h5toNumpy(filenames=[]):
    arrays = list()
    for filename in filenames:
        f = h5py.File(filename, 'r')
        
        spectra_panagia = np.array(f['dataset'])
        energies = np.array(f['energies'])
        f.close()
        arrays.append((spectra_panagia, energies))
        
    return arrays

toTensor = transforms.ToTensor()
input_tensor = toTensor(Image.open('./data/panagia.png'))

w = input_tensor.shape[-2]
h = input_tensor.shape[-1]
r = 6
c = 9
step = 1 

class patch: 
    def __init__(self, arr, r , c):
        self.arr = arr 
        self.r = r
        self.c = c 
    
        
        
for i in range(w):
    for j in range(h):
        
        if(i + r > h and j + c  > w):
            print("both case: ( ", i,":", w, ",",j, ":", h, ")")
        elif(i+r > w):
            print("row case: ( ", i, ":", w, ",",j, ":", j+c, ")")
        elif(j+c > h):
            print("col case: ( ", j, ":", i+r , ",", j, ":", h, ")")
        else:
            print("norm case: ( ", i,":",i+r,",",j,":",j+j, ")")
        input()