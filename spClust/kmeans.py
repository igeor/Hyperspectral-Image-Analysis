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
parser.add_argument("-imS", "--imgShape", nargs="+", default=[21,33,840])
parser.add_argument("-fn", "--fileName", default='panagia.h5')


args = parser.parse_args()
args.n_clusters = [int(n_cluster) for n_cluster in args.n_clusters]
args.imgShape = [int(dim) for dim in args.imgShape]


def h5toNumpy(filepath):
    f = h5py.File(filepath, 'r')
    image = np.array(f['dataset'])
    energies = np.array(f['energies'])
    f.close()
    return image[:,60:900]

X = h5toNumpy('../data/'+args.fileName)

class Pixel :
    def __init__(self,init_i,clust_i,vec):
        self.init_i = init_i
        self.clust_i =clust_i
        self.vec = vec

cmap = { 
    0 : [0,0,0], #black
    1 : [135,34,38], #red,
    2 : [3,4,94], #blue
    3 : [82,183,136], #green,
    4 : [123,44,191], #purple,
    5 : [215,214,10] #yellow
}

imgShape = tuple(args.imgShape)

    
for n_clusters in args.n_clusters:

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    

    # map cluster to color and save as image
    X_out = np.zeros((imgShape[0], imgShape[1], 3))    
    for i, pixel_label in enumerate(reversed(clusterer.labels_)):
        r = i // imgShape[1]; c = i % imgShape[1]
        X_out[r,c,:] = np.array(cmap[pixel_label])

    out_img = Image.fromarray(X_out.astype(np.uint8))
    out_img.save('test'+str(n_clusters)+'.png')


    print('---subclustering---')

    cluster_0 = dict()
    cluster_1 = dict()

    for i, pixel_label in enumerate(reversed(clusterer.labels_)):
        r = i // imgShape[1]; c = i % imgShape[1]
        if( pixel_label == 0):
            cluster_0[i] = Pixel(init_i=i, clust_i=len(cluster_0), vec=X[i])
        else:
            cluster_1[i] = Pixel(init_i=i, clust_i=len(cluster_1), vec=X[i])


    toSubCluster = int(input('Which cluster to subcluster (type 0 or 1)?'))
    nSubclusters = int(input('Type n Subclusters'))

    if(toSubCluster == 1):
        X_1 = np.array([x.vec for x in cluster_1.values()])
        kmeans = KMeans(n_clusters=3, random_state=10).fit(X_1)
    else:
        X_0 = np.array([x.vec for x in cluster_0.values()])
        kmeans = KMeans(n_clusters=3, random_state=10).fit(X_0)

    for i, pixel_label in enumerate(reversed(clusterer.labels_)):
        if( pixel_label == toSubCluster):
            r = i // imgShape[1]; c = i % imgShape[1]
            #print('Pixel (', r, ',', c ,') belongs to cluster 1 and subclustered to cluster',kmeans.labels_[cluster_1[i].clust_i])# ;input()
            
            if(toSubCluster == 1):
                for subClustLabel in range(nSubclusters - 1):
                    if(kmeans.labels_[cluster_1[i].clust_i] == subClustLabel):
                        X_out[r,c,:] = np.array(cmap[subClustLabel + 2])
               
            else:
               for subClustLabel in range(nSubclusters - 1):
                    if(kmeans.labels_[cluster_0[i].clust_i] == subClustLabel):
                        X_out[r,c,:] = np.array(cmap[subClustLabel + 2])
         
    out_img = Image.fromarray(X_out.astype(np.uint8))
    out_img.save('subtest'+str(n_clusters)+'.png')