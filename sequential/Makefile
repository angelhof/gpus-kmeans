CXX=g++
CFLAGS=-O3 -Wall

all: serial_naive run

serial_naive: kmeans.c
	$(CXX) $(CFLAGS) kmeans.c -o kmeans

ready_parallel: kmeans_ready_to_parallel.c
	$(CXX) $(CFLAGS) kmeans_ready_to_parallel.c -o kmeans

run:
	./kmeans < ../data/input.in

clean:
	rm kmeans
