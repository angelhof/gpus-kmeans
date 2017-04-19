import subprocess
import sys

class Result():
    def __init__(self, dataset_name, centers, time, iters):
        self.ds_name = dataset_name
        self.centers = centers
        self.time = time
        self.iters = iters

def find(func, list):
    for el in list:
        if(func(el)):
            return el
    return None


## NUmber of averaged runs
number_of_runs = 10

# Hardcoded datasets relative to data_dir
data_dir = "data/"
datasets = ["iris_dataset/iris.data",
            "elki_sample_dataset/dataset_elki_150.in",
            "elki_sample_dataset/dataset_elki_500.in",
            "road_spatial_network_dataset/spatial_network.data",
            "nu_minebench_dataset/kmeans/color.data",
            "nu_minebench_dataset/kmeans/edge.data"
            "daily_sports_activities/data.data"
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
        all_iters = []
        for run_number in xrange(1, number_of_runs+1):
            ## Print run information
            sys.stdout.write('\r      Run: ' + str(run_number) + '/' + str(number_of_runs))
            sys.stdout.flush()

            ## Execute the program
            ps = [impl, ds]
            proc = subprocess.Popen(
                ps, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (out, err) = proc.communicate()            
            output_lines = out.rstrip().split("\n")
            
            ## Gather results
            raw_iters = find(lambda x: x.startswith("Total num. of steps is"), output_lines)
            iters = int(raw_iters[:-1].split()[-1])

            raw_time = find(lambda x: x.startswith("Total Time Elapsed:"), output_lines)
            real_time = float(raw_time.split()[-2])

            all_iters.append(iters)
            times.append(real_time)

        ## TODO: Also gather centers 
        centers = []

        avg_time = float(sum(times))/len(times)
        avg_iters = float(sum(all_iters))/len(all_iters)
        record[name].append(Result(ds, centers, avg_time, avg_iters))
        sys.stdout.write('\r')
        sys.stdout.flush()

print "                                                   "
# Print the time results
print " ---- RESULTS ----"
for (impl, name) in implementations:
    for res in record[name]:
        print name, ",\t", res.ds_name, ",\t", res.time, "s,\t", int(res.iters), "iters"
