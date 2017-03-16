from random import sample
from math import hypot
import operator
from copy import deepcopy

def distance(p1, p2):
	(x1, y1) = p1
	(x2, y2) = p2
	return hypot(x2-x1, y2-y1)


# read input
line = raw_input().split()
n = int(line[0])
k = int(line[1])
points = [map(float, raw_input().split()) for i in xrange(n)]

# initiate means
means = sample(points, k)
check = 1

# start algorithm
while check>0.001:
	# assign points
	clusters = [[] for i in xrange(k)]
	for point in points:
		distances = [distance(point, m) for m in means]
		mean, min_dist = min(enumerate(distances), key=operator.itemgetter(1))
		clusters[mean].append(point)

	prev_means = deepcopy(means)
	# update means
	for i in xrange(k):
		points_in_cluster = clusters[i]
		if not points_in_cluster:
			continue
		xs, ys = zip(*points_in_cluster)
		number_of_points = len(points_in_cluster)
		means[i] = (sum(xs)/number_of_points, sum(ys)/number_of_points)

	check = sum([distance(prev_means[i], means[i]) for i in xrange(k)])

print '\n'.join(map(str, clusters))

print means