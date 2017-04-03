import matplotlib.pyplot as plt
from random import random

with open("data/dataset_100_5_2_0") as f:
    data = f.readlines()

with open("GPU/centers.out") as f:
    centers_data = f.readlines()

with open("GPU/point_cluster_map.out") as f:
    point_in_centers_data = f.readlines()

# Prepare point data
data = data[1:]  # Skip first description line
for i in range(len(data)):
    l = data[i]
    data[i] = map(float, l.rstrip().split())

# Prepare center data
for i in range(len(centers_data)):
    l = centers_data[i]
    centers_data[i] = map(float, l.rstrip().split())

# Prepare mapping data:
pc_data = []
for i in range(len(point_in_centers_data)):
    # print "pcdata", i, point_in_centers_data[i]
    pc_data.append(int(point_in_centers_data[i].rstrip()))


# Assign points to clusters
cluster_num = len(centers_data)
cluster_list = []
cluster_colors = []

for i in range(cluster_num):
    cluster_list.append([])
    cluster_colors.append((random(), random(), random()))

for i in range(len(pc_data)):
    cluster_list[pc_data[i]].append(i)

# Plot shit
fig = plt.figure()
ax = plt.subplot()

ax.set_title("2D Data Points")
ax.set_xlabel('x')
ax.set_ylabel('y')
# ax.grid(linewidth=2)

# Plot Clusters
for i in range(cluster_num):
    c = centers_data[i]
    cl = cluster_list[i]
    ax.scatter(c[0], c[1], marker='^', s=200, c=cluster_colors[i])
    for j in range(len(cl)):
        p = cl[j]
        ax.scatter(data[p][0], data[p][1], marker='o', c=cluster_colors[i])

plt.show()
