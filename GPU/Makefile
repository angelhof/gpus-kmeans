CXX=g++
# CFLAGS=-O3 -lm -Wall
CUDA_FLAGS=-O3 -D_FORCE_INLINES
CUDA_PREFIX=/opt/cuda-7.5
GPU_COMP?=nvcc
GPU_BACKEND=g++
GPU_FLAGS=--ptxas-options=-v
GPU_ARCH=-arch sm_20
CUBLAS=-lcublas
CUBLAS_PATH=-L$(CUDA_PREFIX)/lib64

all: kmeans_gpu run


kmeans_util.o: kmeans_util.cu
	$(GPU_COMP) $(CUDA_FLAGS) -c kmeans_util.cu $(GPU_FLAGS)

gpu_util.o: gpu_util.cu
	$(GPU_COMP) $(CUDA_FLAGS) -c gpu_util.cu $(GPU_FLAGS)

test: gpu_util.o kmeans_gpu_not_atomic.cu 
	nvcc -O3 -D_FORCE_INLINES kmeans_gpu_not_atomic.cu -c -L/opt/cuda-7.5/lib64 -lcublas
	nvcc -O3 -D_FORCE_INLINES kmeans_gpu_not_atomic.o gpu_util.o -o kmeans_gpu -L/opt/cuda-7.5/lib64 -lcublas

kmeans_gpu: gpu_util.o kmeans_util.o kmeans_gpu_not_atomic.cu
	$(GPU_COMP) $(CUDA_FLAGS) -c kmeans_gpu_not_atomic.cu -o kmeans_gpu.o
	$(GPU_COMP) $(CUDA_FLAGS) kmeans_gpu.o gpu_util.o kmeans_util.o -o kmeans_gpu $(CUBLAS_PATH) $(CUBLAS)

check_architectures: check_architecture.cu
	$(GPU_COMP) $(CUDA_FLAGS) check_architecture.cu -o get_arch
	./get_arch

run:
	# ./kmeans_gpu 256 "../data/road_spatial_network_dataset/spatial_network.data"
	#./kmeans_gpu 32 "../data/dataset_100_5_2_0"
	./kmeans_gpu 256 ../data/input.in
	# ./kmeans_gpu < ../data/input.in
	
clean:
	rm ./kmeans_gpu *.o