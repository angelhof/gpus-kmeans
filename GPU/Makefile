CXX=g++
CFLAGS=-O3 -lm -Wall
CUDA_PREFIX = /usr/local/cuda
GPU_COMP ?= nvcc
GPU_BACKEND=g++
GPU_FLAGS=-ccbin
GPU_ARCH=-arch sm_20

all: kmeans_gpu run

kmeans_gpu: kmeans_gpu_not_atomic.cu gpu_util.cu
	$(GPU_COMP) -c kmeans_gpu_not_atomic.cu
	$(GPU_COMP) -c gpu_util.cu
	$(GPU_COMP) kmeans_gpu_not_atomic.o gpu_util.o -o kmeans_gpu
	

check_architectures: check_architecture.cu
	$(GPU_COMP) check_architecture.cu -o get_arch
	./get_arch

run:
	./kmeans_gpu < ../data/road_spatial_network_dataset/spatial_network.data

clean:
	rm kmeans 
