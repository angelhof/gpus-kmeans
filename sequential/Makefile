CXX=g++
CFLAGS=-O3 -lm -Wall

all: serial_naive run

serial_naive: kmeans.c
	$(CXX) $(CFLAGS) kmeans.c -o kmeans

run:
	./kmeans < ../data/input.in

clean:
	rm kmeans
