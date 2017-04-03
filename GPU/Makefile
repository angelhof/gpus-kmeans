CXX=g++
CFLAGS=-O3 -lm -Wall
CUDA_PREFIX = /opt/cuda-7.5/bin/nvcc
GPU_COMP ?= /opt/cuda-7.5/bin/nvcc
GPU_BACKEND=g++
GPU_FLAGS=-ccbin
GPU_ARCH=-arch sm_20

CUBLAS=-lcublas
CUBLAS_PATH=-L/opt/cuda-7.5/lib64

all: kmeans_gpu run

kmeans_gpu: kmeans_gpu_not_atomic.cu gpu_util.cu
	$(GPU_COMP) -c kmeans_gpu_not_atomic.cu -o kmeans_gpu.o
	$(GPU_COMP) -c gpu_util.cu
	$(GPU_COMP) kmeans_gpu.o gpu_util.o -o kmeans_gpu $(CUBLAS_PATH) $(CUBLAS)


check_architectures: check_architecture.cu
	$(GPU_COMP) check_architecture.cu -o get_arch
	./get_arch

run:
	./kmeans_gpu 32 "../data/dataset_100_5_2_0"
	#./kmeans_gpu 32 "../data/input.in"
	# ./kmeans_gpu < ../data/input.in
	# ./kmeans_gpu < ../data/road_spatial_network_dataset/spatial_network.data

clean:
	rm kmeans_gpu *.o

