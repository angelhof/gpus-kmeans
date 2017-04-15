
in_file = "data_original.data"
out_file = "data.data"

# The exercise types
n_clusters = 19
n_attrs = 45


with open(in_file, "r") as fin:
    data = fin.read().rstrip().split("\n")
    n_vectors = len(data)


with open(out_file, "w") as fout:
    fout.write(str(n_vectors) + " " + str(n_clusters) +
               " " + str(n_attrs) + "\n")

    for vector in data:
        line = " ".join(vector.split(","))
        fout.write(line + "\n")
