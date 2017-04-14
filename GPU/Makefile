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



kmeans_util.o: kmeans_util_sa.cu
	$(GPU_COMP) $(CUDA_FLAGS) -c kmeans_util_sa.cu $(GPU_FLAGS) -o kmeans_util.o

gpu_util.o: gpu_util.cu
	$(GPU_COMP) $(CUDA_FLAGS) -c gpu_util.cu $(GPU_FLAGS)

gpu_minibatch_util.o: kmeans_minibatch_gpu.cu kmeans_util.o
	$(GPU_COMP) $(CUDA_FLAGS) -c kmeans_minibatch_gpu.cu $(GPU_FLAGS) -o gpu_minibatch_util.o

gpu_sa_util.o: kmeans_sa_gpu.cu kmeans_util.o
	$(GPU_COMP) $(CUDA_FLAGS) -c kmeans_sa_gpu.cu $(GPU_FLAGS) -o gpu_sa_util.o


kmeans_gpu: gpu_util.o kmeans_util.o gpu_minibatch_util.o gpu_sa_util.o kmeans_gpu_not_atomic_sa.cu
	$(GPU_COMP) $(CUDA_FLAGS) -c kmeans_gpu_not_atomic_sa.cu -o kmeans_gpu.o
	$(GPU_COMP) $(CUDA_FLAGS) gpu_minibatch_util.o gpu_sa_util.o kmeans_gpu.o gpu_util.o kmeans_util.o -o kmeans_gpu $(CUBLAS_PATH) $(CUBLAS)

check_architectures: check_architecture.cu
	$(GPU_COMP) $(CUDA_FLAGS) check_architecture.cu -o get_arch
	./get_arch

run:
	# ./kmeans_gpu 256 "../data/road_spatial_network_dataset/spatial_network.data"
	./kmeans_gpu 32 "../data/dataset_100_5_2_0"
	# ./kmeans_gpu 32 "../data/input.in"
	# ./kmeans_gpu < ../data/input.in
	
clean:
	rm ./kmeans_gpu *.o *out