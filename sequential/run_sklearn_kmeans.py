from sklearn.cluster import KMeans
import numpy as np

with open("input.in") as f:
    data = f.readlines()

data_lines = data[1:-1]
X = map(lambda x: map(float, x.split()), data_lines)
# X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print kmeans.labels_
print kmeans.cluster_centers_
