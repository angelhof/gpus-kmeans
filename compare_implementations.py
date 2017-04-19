import subprocess
import sys

class Result():
    def __init__(self, dataset_name, centers, time):
        self.ds_name = dataset_name
        self.centers = centers
        self.time = time


## NUmber of averaged runs
number_of_runs = 10

# Hardcoded datasets relative to data_dir
data_dir = "data/"
datasets = ["iris_dataset/iris.data",
            "elki_sample_dataset/dataset_elki_150.in",
            "elki_sample_dataset/dataset_elki_500.in",
            # "road_spatial_network_dataset/spatial_network.data",
            "nu_minebench_dataset/kmeans/color.data",
            "nu_minebench_dataset/kmeans/edge.data"
            # "daily_sports_activities/data.data"
            ]

datasets = map(lambda x: data_dir + x, datasets)

# Implementations
implementations = [
    ("./serial/run_sklearn_kmeans.py", "scikit_kmeans"),
    # ("./GPU/kmeans_cublas 128 ./data/input.in", "cublas"),
    # ("./GPU/kmeans_cublas_sa 128 ./data/input.in", "cublas_simulated_annealing"),
    # ("./GPU/kmeans_reduce 128 ./data/input.in", "reduce"),
    # ("./GPU/kmeans_cusparse 128 2 ./data/input.in", "cusparse")
    ]

# Create a dictionary with all implementations
# that will contain a list of results
record = {}
for (impl, name) in implementations:
    record[name] = []

# Run all implementations
for (impl, name) in implementations:
    for ds in datasets:
        print " -- Executing: " + name + " - " + ds
        times = []
        for run_number in xrange(1, number_of_runs+1):
            ## Print run information
            sys.stdout.write('\r      Run: ' + str(run_number) + '/' + str(number_of_runs))
            sys.stdout.flush()

            ## Execute the program
            ps = ["time", "-p", impl, ds]
            proc = subprocess.Popen(
                ps, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (out, err) = proc.communicate()
            centers_str = out.rstrip().split("\n")

            ## Gather the results
            centers = [map(float, line.split()) for line in centers_str]
            user_time = float(err.split("\n")[1].split(" ")[1])
            real_time = float(err.split("\n")[0].split(" ")[1])
            
            times.append(real_time)

        avg_time = float(sum(times))/len(times)
        record[name].append(Result(ds, centers, avg_time))
        sys.stdout.write('\r')
        sys.stdout.flush()

print "                                                   "

# Print the time results
print " ---- RESULTS ----"
for (impl, name) in implementations:
    for res in record[name]:
        print name, "\t", res.ds_name, "\t", res.time, "s"
