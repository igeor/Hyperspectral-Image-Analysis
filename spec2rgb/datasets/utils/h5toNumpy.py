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
