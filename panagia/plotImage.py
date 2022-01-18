import h5py
import numpy as np
import os.path 
from matplotlib import pyplot as plt

from datasets.utils import h5toNumpy

f = h5py.File(os.path.join( os.getcwd(), 'panagia\data\\', 'panagia.h5' ), 'r')

spectra_panagia = np.array(f['dataset'])
spectra_panagia = spectra_panagia.reshape(21,33,2048)
img = np.sum(spectra_panagia[:,:,1000], axis=2)
#img = spectra_panagia[:,:,780]
img = np.rot90(img);img = np.rot90(img)

plt.imshow(img)
plt.show()
