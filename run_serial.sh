#!/bin/bash

## Use "qsub -q termis -l nodes=termis:cuda run.sh" to run script

## output file
#PBS -o run_serial.out
## error file
#PBS -e run_serial.err
## expected runtime
#PBS -l walltime=01:00:00
#PBS -l nodes=termi3:cuda


export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

## program name
gpu_prog="./kmeans"
input="/home/parallel/parlab18/GPUs/data/iris_dataset/iris.data"

## go to your directory
cd /home/parallel/parlab18/GPUs/sequential
module load gcc/4.8.2
make clean
## Kalo einai na kaneis make panw sto node
make

## run program
time $gpu_prog < $input
