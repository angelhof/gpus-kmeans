#!/bin/bash

# Download the dataset
# wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00256/data.zip"

# Combine all files into one
find . -type f -name '*.txt' -exec cat {} + >> data_original.data
