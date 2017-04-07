import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

with open(sys.argv[1]) as f:
    n, k, dim= map(int, f.readline().split())
    X = np.loadtxt(f)

kmeans = KMeans(n_clusters=k, random_state=0, n_jobs=-1, n_init=1).fit(X)
print "Simple K-Means clustering with K-Means++ initialization:"
print "-Labels:"
print kmeans.labels_
print "-Centers:"
print kmeans.cluster_centers_

# minibatch = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=1).fit(X)
# print "\nMini-Batch K-Means clustering with K-Means++ initialization:"
# print "-Labels:"
# print minibatch.labels_
# print "-Centers:"
# print minibatch.cluster_centers_
