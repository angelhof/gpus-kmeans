CXX=g++
CFLAGS=-O3 -lm -Wall

all: gpu_naive run

gpu_naive: kmeans_gpu.cu
	$(CXX) $(CFLAGS) kmeans_gpu.cu -o kmeans

run:
	./kmeans < ../data/input.in

clean:
	rm kmeans
