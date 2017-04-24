"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt

def parse_results(condition, filename):
	fin = open(filename)
	lines = fin.read().rstrip().split("\n")
	fin.close()
	kept_lines = filter(condition, lines)
	result = map(lambda x: float(x.split(",")[-2].split()[0]), kept_lines)
	return result

def plot_one(output_name, dataset, files, x_labels):
	cond = lambda x: x.startswith('cusparse, ' + dataset + ',')
	cusparse_times = parse_results(cond, files[0])
	N = len(cusparse_times)

	cond = lambda x: x.startswith('cublas, ' + dataset + ',')
	cublas_times = parse_results(cond, files[1])


	cond = lambda x: x.startswith('scikit_kmeans, ' + dataset + ',')
	serial_times = parse_results(cond, files[2])


	# 

	max_value = max(max(serial_times), max(cublas_times), max(cusparse_times))

	## TODO: Put inertias here

	ind = np.arange(N)  # the x locations for the groups
	width = 0.20       # the width of the bars
	gap = 0.05
	n_y_ticks = 10

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, serial_times, width, color='g')
	rects2 = ax.bar(ind + (width + gap), cublas_times, width, color='c')
	rects3 = ax.bar(ind + 2*(width + gap), cusparse_times, width, color='m')

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Time (seconds)')
	ax.set_title('Time of execution for dataset x')
	ax.set_xticks(ind + width / 2)
	ax.set_xticklabels(x_labels)
	# ax.set_yticks(np.arange(0, max_value * 1.1, max_value/n_y_ticks))
	ax.set_yscale('log')

	ax.legend((rects1[0], rects2[0], rects3[0]), ('serial', 'cuBlas', 'cuSparse'))
	ax.grid(True)
	plt.savefig(output_name)	

## TODO: Show inertia somewhere

output_name = "road_dataset.png"
dataset = 'data/road_spatial_network_dataset/spatial_network.data'
files = ["titan_x_final.txt"] * 2 + ["scikit_final.out"]
x_labels = map(str, range(5,45,5) + [55])
plot_one(output_name, dataset, files, x_labels)

output_name = "nu_minebench.png"
dataset = 'data/nu_minebench_dataset/kmeans/edge.data'
files = ["titan_x_final.txt"] * 2 + ["scikit_final.out"]
x_labels = map(str, range(50,400,50) + [500, 600])
plot_one(output_name, dataset, files, x_labels)

