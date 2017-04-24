#!/usr/bin/env python
import sys
import numpy as np
from sklearn.cluster import KMeans
# from sklearn.cluster import MiniBatchKMeans
from datetime import datetime

if(len(sys.argv) == 1):
    text = sys.stdin.readlines()
    params = map(int, text[0].split(" "))
    n, k, dim = params[0], params[1], params[2]
    x_list = []
    for line in text[1:]:
        x_list.append(np.fromstring(line, dtype=float, sep=' '))
    X = np.array(x_list)
else:
    k = int(sys.argv[1])
    with open(sys.argv[2]) as f:
        n, _, dim = map(int, f.readline().split())
        X = np.loadtxt(f)

start_time = datetime.now()

kmeans = KMeans(n_clusters=k,  
                n_init=1, 
                # init='random',
                precompute_distances=False,
                tol=1e-9,
                max_iter=10000).fit(X)

end_time = datetime.now()
# print "Simple K-Means clustering with K-Means++ initialization:"
# print "-Labels:"
# print kmeans.labels_
# print "-Centers:"

steps = kmeans.n_iter_
duration = end_time - start_time
seconds_elapsed = duration.total_seconds()
print "Total num. of steps is %d." % (steps)
print "Total Time Elapsed: %lf seconds" % (seconds_elapsed)
print "Time per step is %lf" % (seconds_elapsed / steps)

print "Centers:"
print '\n'.join(' '.join('%f' % x for x in y) for y in kmeans.cluster_centers_)

print "Sum of distances of samples to their closest cluster center: %lf" % (kmeans.inertia_)

# minibatch = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=1).fit(X)
# print "\nMini-Batch K-Means clustering with K-Means++ initialization:"
# print "-Labels:"
# print minibatch.labels_
# print "-Centers:"
# print minibatch.cluster_centers_
