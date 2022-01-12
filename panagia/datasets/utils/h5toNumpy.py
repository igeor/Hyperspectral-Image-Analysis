import h5py
import numpy as np
import os.path 


'''
@exports:   
    a list of tuples that contain 
    spectra-image and energies 
'''
def h5toNumpy(filenames=[]):
    arrays = list()
    for filename in filenames:
        f = h5py.File(os.path.join( os.getcwd(), 'panagia\data\\', filename ), 'r')
        
        spectra_panagia = np.array(f['dataset'])
        energies = np.array(f['energies'])
        f.close()
        arrays.append((spectra_panagia, energies))
        
    return arrays
