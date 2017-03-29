#!/bin/bash

# Download dataset and description
wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data 
wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names

# Rename the data file
mv iris.data iris_original.data

# Prepare it for the kmeans program
python iris_dataset_transformer.py
