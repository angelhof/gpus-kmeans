#!/usr/bin/python

# The original file
dataset_file = ['spatial_network', '_original', '.data']

# This dataset doesn't have a correct cluster number so try with different ones
k_clusters = 5

file_name = dataset_file
# Open file and read all lines  
f_in = open("".join(file_name))  
lines = f_in.read().rstrip().split('\n')
f_in.close()

# Remove the label from the lines
data_lines = map(lambda x: x.split(',')[1:], lines)

# Total data points
n = len(data_lines)

# Write to file
f_out = open(file_name[0] + file_name[2], 'w') 

# Write constants
f_out.write(str(n) + " " + str(k_clusters) + "\n")

for line in data_lines:
  f_out.write(" ".join(line) + "\n")

f_out.close()
