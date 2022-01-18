from __init__ import *

f = h5py.File( parentdir + "\\" + "data\\jesus.h5", 'r')

spectra_panagia = np.array(f['dataset'])
spectra_panagia = spectra_panagia.reshape(31,46,2048)
img = np.sum(spectra_panagia[:,:,:], axis=2)

img = npRotate(img, deg=180)

plt.imshow(img)
plt.show()
