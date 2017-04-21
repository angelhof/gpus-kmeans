import subprocess
import sys


class Result():
    def __init__(self, dataset_name, centers, k, time, iters, time_per_iter, inertia):
        self.ds_name = dataset_name
        self.centers = centers
        self.k = k
        self.time = time
        self.iters = iters
        self.time_per_iter = time_per_iter
        self.inertia = inertia


def find(func, list):
    for el in list:
        if(func(el)):
            return el
    return None


# NUmber of averaged runs
number_of_runs = 5

# Hardcoded datasets relative to data_dir
data_dir = "data/"
datasets = [
    # "iris_dataset/iris.data",
    # "elki_sample_dataset/dataset_elki_150.in",
    # "elki_sample_dataset/dataset_elki_500.in",
    "road_spatial_network_dataset/spatial_network.data",
    # "nu_minebench_dataset/kmeans/color.data",
    "nu_minebench_dataset/kmeans/edge.data",
    "daily_sports_activities/data.data"
]

datasets = map(lambda x: data_dir + x, datasets)

# Implementations
implementations = [
    # ("./serial/run_sklearn_kmeans.py", "scikit_kmeans"),
    # ("./serial/kmeans", "serial"),
    ("./GPU/kmeans_cublas", "cublas"),
    ("./GPU/kmeans_cublas_sa", "cublas_simulated_annealing"),
    # ("./GPU/kmeans_reduce", "reduce"),  # Something wrong here
    ("./GPU/kmeans_cusparse", "cusparse")
]

# Different values for k
k_values = {
    "iris_dataset/iris.data": ["3", "6", "10"],
    "elki_sample_dataset/dataset_elki_150.in": ["3", "6", "10"],
    "elki_sample_dataset/dataset_elki_500.in": ["3", "10", "20"],
    "road_spatial_network_dataset/spatial_network.data": ["5", "10", "15", "20", "25", "30", "35", "40", "45", "55"],
    "nu_minebench_dataset/kmeans/color.data": ["200", "300", "400"],
    "nu_minebench_dataset/kmeans/edge.data": ["50", "100", "150", "200", "250", "300", "350", "400", "500", "600"],
    "daily_sports_activities/data.data": ["5", "8", "10", "13", "15", "18", "20", "25", "30", "35"]
}

# Create a dictionary with all implementations
# that will contain a list of results
record = {}
for (impl, name) in implementations:
    record[name] = []

# Run all implementations
for (impl, name) in implementations:
    for ds in datasets:
        for k in k_values[ds[5:]]:
            print " -- Executing: " + name + " - " + ds + " - k = " + k
            times = []
            all_iters = []
            times_per_iter = []
            inertias = []
            for run_number in xrange(1, number_of_runs + 1):
                # Print run information
                sys.stdout.write('\r      Run: ' +
                                 str(run_number) + '/' + str(number_of_runs))
                sys.stdout.flush()

                # Execute the program
                ps = [impl, k, ds]
                if name == "scikit_kmeans":
                    ps = [impl, k, ds]
                elif name == "serial":
                    ps = [impl, k, ds]
                elif name == "cublas":
                    ps = [impl, "256", k, ds]
                elif name == "cublas_simulated_annealing":
                    ps = [impl, "256", k, ds]
                elif name == "reduce":
                    ps = [impl, "256", k, ds]
                elif name == "cusparse":
                    if ds == "road_spatial_network_dataset/spatial_network.data":
                        ps = [impl, "256", "1024", k, ds]
                    elif ds == "daily_sports_activities/data.data":
                        ps = [impl, "64", "4096", k, ds]
                    elif "nu_minebench_dataset/kmeans" in ds:
                        ps = [impl, "64", "1024", k, ds]
                    else:
                        ps = [impl, "128", "1024", k, ds]

                try:
                    proc = subprocess.Popen(
                        ps, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    (out, err) = proc.communicate()
                    output_lines = out.rstrip().split("\n")

                    # Gather results
                    raw_iters = find(lambda x: x.startswith(
                        "Total num. of steps is"), output_lines)
                    iters = int(raw_iters[:-1].split()[-1])

                    raw_time = find(lambda x: x.startswith(
                        "Total Time Elapsed:"), output_lines)
                    real_time = float(raw_time.split()[-2])

                    raw_time_per_step = find(lambda x: x.startswith(
                        "Time per step is"), output_lines)
                    real_time_per_step = float(raw_time_per_step.split()[-1])

                    raw_inertia = find(lambda x: x.startswith(
                        "Sum of distances of samples to their closest cluster center:"), output_lines)
                    real_inertia = float(raw_inertia.split()[-1])

                    all_iters.append(iters)
                    times.append(real_time)
                    times_per_iter.append(real_time_per_step)
                    inertias.append(real_inertia)
                except KeyboardInterrupt:
                    print "\nExiting..."
                    exit(1)
                except:
                    print "\rUnexpected error:", sys.exc_info()[0]

            # TODO: Also gather centers
            centers = []

            try:
                avg_time = float(sum(times)) / len(times)
                avg_iters = float(sum(all_iters)) / len(all_iters)
                avg_time_per_iter = float(
                    sum(times_per_iter)) / len(times_per_iter)
                avg_inertia = float(sum(inertias)) / len(inertias)
                record[name].append(Result(ds, centers, k, avg_time,
                                           avg_iters, avg_time_per_iter,
                                           avg_inertia))
            except KeyboardInterrupt:
                print "\nExiting..."
                exit(1)
            except:
                print "\rUnexpected error:", sys.exc_info()[0]
            sys.stdout.write('\r')
            sys.stdout.flush()

print "                                                   "
# Print the time results
fout = open("results.out", "w")
print " ---- RESULTS ----"
for (impl, name) in implementations:
    for res in record[name]:
        out_str = "{}, {}, k = {}, {:0.8f} sec, {} iters, {:0.8f} sec/iter, {} Inertia"
        out_str = out_str.format(name, res.ds_name, res.k,
                                 res.time, int(res.iters), res.time_per_iter,
                                 res.inertia)
        print out_str
        out_str += "\n"
        fout.write(out_str)
fout.close()
