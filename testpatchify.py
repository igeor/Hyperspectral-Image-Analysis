import image_slicer
from PIL import ImageDraw, ImageFont, Image
import numpy as np 
import torch 
import numpy as np
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt

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


npSpec = h5toNumpy(['./data/jesus.h5'])[0][0]
npSpec = npSpec.reshape((31,46,2048))
npSpecsum = np.sum(npSpec[:,:,:], axis=2)

print(npSpec.shape)

npimg = np.asarray(Image.open('./data/jesus.png'))
print(npimg.shape)
 
patchesRGB = patchify(npimg, (15,15,3), step=1) # split image into 2*3 small 2*2 patchesRGB.
patchesSPEC = patchify(npSpec, (15,15,2048), step=1) # split image into 2*3 small 2*2 patchesRGB.


num_patchesRGB = 0
num_rows = 0
for patch in patchesRGB:
    num_rows += 1
    for row in range(patch.shape[0]):
        num_patchesRGB += 1
        #im = Image.fromarray(patch[row,0,:,:,:])
        #im.save('./results/t'+str(num_patchesRGB)+'-'+str(num_rows)+'.png')
reconstructed_image = unpatchify(patchesRGB, (31,46,3))
imr = Image.fromarray(reconstructed_image)
imr.save('./results/outttt.png')

num_patchesSPEC = 0
num_rows = 0
for patch in patchesSPEC:
    num_rows += 1
    for row in range(patch.shape[0]):
        num_patchesSPEC += 1
        #im = Image.fromarray(patch[row,0,:,:,:])
        #im.save('./results/t'+str(num_patchesSPEC)+'-'+str(num_rows)+'.png')     


reconstructed_image_SPEC = unpatchify(patchesSPEC, (31,46,2048))
imrSPEC = Image.fromarray(np.sum(reconstructed_image_SPEC[:,:,:], axis=2)).convert('RGB')
print(imrSPEC.size, imr.size)
imrSPEC.save('./results/outttt2.png')


imgplot = plt.imshow(np.sum(npSpec[:,:,:], axis=2))
plt.show()

imgplot = plt.imshow(np.sum(reconstructed_image_SPEC[:,:,:], axis=2))
plt.show()