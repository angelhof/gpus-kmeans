#!/usr/bin/python

# The original elki files
elki_files = [ ['dataset_elki_150', '_original', '.in']
             , ['dataset_elki_500', '_original', '.in']]

k_clusters = 3

for file_name in elki_files:

  # Open file and read all lines  
  f_in = open("".join(file_name))  
  lines = f_in.read().rstrip().split('\n')
  f_in.close()

  # Keep the lines that don't start with #
  data_lines_with_label = filter(lambda x: not x.startswith('#'), lines)

  # Remove the label from the lines
  data_lines = map(lambda x: x.split()[:2], data_lines_with_label)
  
  # Total data points
  n = len(data_lines)

  # Write to file
  f_out = open(file_name[0] + file_name[2], 'w') 

  # Write constants
  f_out.write(str(n) + " " + str(k_clusters) + "\n")
  
  for line in data_lines:
    f_out.write(" ".join(line) + "\n")

  f_out.close()
