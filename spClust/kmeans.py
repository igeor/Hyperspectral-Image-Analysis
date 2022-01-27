from audioop import reverse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import numpy as np
import h5py 
import argparse 

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("-cl", "--n_clusters", nargs="+", default=[2,3,4,5,6])
args = parser.parse_args()
args.n_clusters = [int(n_cluster) for n_cluster in args.n_clusters]


def h5toNumpy(filepath):
    f = h5py.File(filepath, 'r')
    image = np.array(f['dataset'])
    energies = np.array(f['energies'])
    f.close()
    return image[:,60:900]

X = h5toNumpy('../data/panagia.h5')

range_n_clusters = [2, 3, 4, 5, 6]
cmap = { 
    0 : [0,0,0], #black
    1 : [135,34,38], #red,
    2 : [3,4,94], #blue
    3 : [82,183,136], #green,
    4 : [123,44,191], #purple,
    5 : [215,214,10] #yellow
}

imgShape = (21, 33, 840)


for n_clusters in args.n_clusters:

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(16, 9)

    '''
    * The 1st subplot is the silhouette plot
    '''

    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [0, 1]
    ax1.set_xlim([0, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    X_out = np.zeros((imgShape[0], imgShape[1], 3))
    
    
    for i, pixel_label in enumerate(reversed(clusterer.labels_)):
        r = i // imgShape[1]
        c = i % imgShape[1]
        
        X_out[r,c,:] = np.array(cmap[pixel_label])
    
    
    out_img = Image.fromarray(X_out.astype(np.uint8))
    out_img.save('test'+str(n_clusters)+'.png')