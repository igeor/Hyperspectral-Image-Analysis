import h5py
import numpy as np
import os.path

root_dir = os.path.dirname(__file__)

'''
h5 files needs to follow 
the below file structure

data
|-- <dataset dir>
|   `-- <dataset name>.py
|   `-- <filename1>.h5
|   `-- <filename2>.h5
|   `-- ...
'''

'''
@exports:   
    a list of tuples that contain 
    spectra-image and energies 
'''
def h5toNumpy(filenames=[]):
    arrays = list()
    for filename in filenames:
        f = h5py.File(
            'data/'+filename, 'r')
        spectra_panagia = np.array(f['dataset'])
        energies = np.array(f['energies'])
        f.close()
        arrays.append((spectra_panagia, energies))
        
    return arrays
