CXX=g++
# CFLAGS=-O3 -lm -Wall
CUDA_FLAGS=-O3 -D_FORCE_INLINES
CUDA_PREFIX=/opt/cuda-7.5
GPU_COMP?=nvcc
GPU_BACKEND=g++
GPU_FLAGS=--ptxas-options=-v
GPU_ARCH=-arch sm_20
CUBLAS=-lcublas
CUSPARSE=-lcusparse
CUBLAS_PATH=-L$(CUDA_PREFIX)/lib64

all: kmeans_cublas kmeans_cublas_sa kmeans_reduce kmeans_cusparse

gpu_util.o: gpu_util.cu
	$(GPU_COMP) $(CUDA_FLAGS) -c gpu_util.cu $(GPU_FLAGS)

kmeans_util.o: kmeans_util.cu
	$(GPU_COMP) $(CUDA_FLAGS) -c kmeans_util.cu $(GPU_FLAGS)

kmeans_cublas: gpu_util.o kmeans_util.o kmeans_gpu_not_atomic.cu
	$(GPU_COMP) $(CUDA_FLAGS) -c kmeans_gpu_not_atomic.cu -o kmeans_cublas.o
	$(GPU_COMP) $(CUDA_FLAGS) kmeans_cublas.o gpu_util.o kmeans_util.o -o kmeans_cublas $(CUBLAS_PATH) $(CUBLAS)

kmeans_util_sa.o: kmeans_util_sa.cu
	$(GPU_COMP) $(CUDA_FLAGS) -c kmeans_util_sa.cu $(GPU_FLAGS)

kmeans_cublas_sa: gpu_util.o kmeans_util_sa.o kmeans_gpu_not_atomic_sa.cu
	$(GPU_COMP) $(CUDA_FLAGS) -c kmeans_gpu_not_atomic_sa.cu -o kmeans_cublas_sa.o
	$(GPU_COMP) $(CUDA_FLAGS) kmeans_cublas_sa.o gpu_util.o kmeans_util_sa.o -o kmeans_cublas_sa $(CUBLAS_PATH) $(CUBLAS)

kmeans_reduce: gpu_util.o kmeans_gpu_reduce.cu
	$(GPU_COMP) $(CUDA_FLAGS) -c kmeans_gpu_reduce.cu -o kmeans_reduce.o
	$(GPU_COMP) $(CUDA_FLAGS) kmeans_reduce.o gpu_util.o -o kmeans_reduce $(CUBLAS_PATH) $(CUBLAS)

kmeans_util_cusparse.o: kmeans_util_cusparse.cu
	$(GPU_COMP) $(CUDA_FLAGS) $(CUSPARSE) -c kmeans_util_cusparse.cu $(GPU_FLAGS)

kmeans_cusparse: gpu_util.o kmeans_util_cusparse.o kmeans_gpu_cusparse.cu
	$(GPU_COMP) $(CUDA_FLAGS) -c kmeans_gpu_cusparse.cu -o kmeans_cusparse.o
	$(GPU_COMP) $(CUDA_FLAGS) kmeans_cusparse.o gpu_util.o kmeans_util_cusparse.o -o kmeans_cusparse $(CUBLAS_PATH) $(CUBLAS) $(CUSPARSE)

check_architectures: check_architecture.cu
	$(GPU_COMP) $(CUDA_FLAGS) check_architecture.cu -o get_arch
	./get_arch

test_run:
	./kmeans_cublas 128 ../data/input.in
	./kmeans_cublas_sa 128 ../data/input.in
	./kmeans_reduce 128 ../data/input.in
	./kmeans_cusparse 128 2 ../data/input.in
	# ./kmeans_gpu 256 "../data/road_spatial_network_dataset/spatial_network.data"
	#./kmeans_gpu 32 "../data/dataset_100_5_2_0"
	# ./kmeans_gpu < ../data/input.in
	
clean:
	rm ./kmeans_cublas ./kmeans_cublas_sa ./kmeans_reduce ./kmeans_cusparse *.o