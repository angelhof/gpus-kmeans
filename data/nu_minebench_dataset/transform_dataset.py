import struct

in_files = [ 'kmeans/color'
		   , 'kmeans/edge']

'''
TODO: 
    1. Run the Nu-Minebench algorithm in order to see the correct number of clusters
    2. It didn't work, it doesnt give back the best number of clusters
'''
## This is not correct , I dont know the correct cluster number
n_clusters = 20

for fin_name in in_files:

	fin = open(fin_name, "rb")
	n_vectors, = struct.unpack('i', fin.read(4))
	n_attrs, = struct.unpack('i', fin.read(4))
	# print n_vectors, n_attrs

	data = []
	for vec in xrange(n_vectors):
		data.append([])
		for attr in xrange(n_attrs):
			temp, = struct.unpack('f', fin.read(4))
			data[vec].append(temp) 
	fin.close()

	fout = open(fin_name+".data", "w")

	fout.write(str(n_vectors) + " " + str(n_clusters) + " " + str(n_attrs) + "\n")

	for vector in data:
		line = " ".join(map(str, vector))
		fout.write(line + "\n")
