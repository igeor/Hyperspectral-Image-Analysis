
from distutils.log import error
import numpy as np
from matplotlib import pyplot as plt

def npRotate(arr, deg=90):
    if(deg not in [90,180,270]):
        raise ValueError('deg not in [90,180,270]')

    if(deg==90): arr = np.rot90(arr); return arr
    if(deg==180): arr = np.rot90(arr);arr = np.rot90(arr); return arr
    if(deg==270): arr = np.rot90(arr);arr = np.rot90(arr);arr = np.rot90(arr); return arr
    
