CXX=g++
CFLAGS=-O3 -lm -Wall
GPU_COMP=nvcc
GPU_BACKEND=g++
GPU_FLAGS=-ccbin
GPU_ARCH=-arch sm_20

all: gpu_naive run

gpu_naive: kmeans_gpu.cu
	$(CXX) $(CFLAGS) kmeans_gpu.cu -o kmeans

check_architectures: check_architecture.cu
	$(GPU_COMP) check_architecture.cu -o get_arch

run:
	./kmeans < ../data/input.in

clean:
	rm kmeans
