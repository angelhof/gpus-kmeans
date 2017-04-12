import subprocess
import os

class Result():
    def __init__(self, dataset_name, centers, time):
        self.ds_name = dataset_name
        self.centers = centers
        self.time = time


## Hardcoded datasets relative to data_dir
data_dir = "data/"
datasets = [ "iris_dataset/iris.data"
           , "elki_sample_dataset/dataset_elki_150.in"
           , "elki_sample_dataset/dataset_elki_500.in"
           , "road_spatial_network_dataset/spatial_network.data"
           , "nu_minebench_dataset/kmeans/color.data"
           , "nu_minebench_dataset/kmeans/edge.data"
           ]

datasets = map(lambda x: data_dir + x, datasets)

## Implementations
implementations = [ ("./sequential/run_sklearn_kmeans.py", "scikit_kmeans")
                  ]

## Create a dictionary with all implementations 
## that will contain a list of results
record = {}
for (impl, name) in implementations:
    record[name] = []

## Run all implementations
for (impl, name) in implementations:
    for ds in datasets:
        ps = ["time", "-p", impl, ds]
        proc = subprocess.Popen(ps, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = proc.communicate()
        centers_str = out.rstrip().split("\n")
        
        centers = [map(float, line.split()) for line in centers_str]
        # print centers
        
        user_time = float(err.split("\n")[1].split(" ")[1])
        real_time = float(err.split("\n")[0].split(" ")[1])
        # print float(user_time)
        # output = os.popen("time -p %s %s" %(impl, ds)).read()
        # print output

        record[name].append(Result(ds, centers, real_time))


## Print the time results
for (impl, name) in implementations:
    for res in record[name]:
        print name,"\t",res.ds_name,"\t", res.time,"s"
