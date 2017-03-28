#!/bin/bash

# Use "qsub -q termis -l nodes=termis:cuda run.sh" to run script

# output file
#PBS -o run.out
# error file
#PBS -e run.err
# expected runtime
#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=1:cuda


export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# program name
gpu_prog="./kmeans_gpu_clean"

# go to your directory
cd /to/your/directory
make clean
# Kalo einai na kaneis make panw sto node
make
echo "Benchmark started on $(date) in $(hostname)"

# run program
$gpu_prog

echo "Benchmark ended on $(date) in $(hostname)"
