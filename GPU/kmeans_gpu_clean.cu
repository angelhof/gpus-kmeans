#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "gpu_util.h"

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#ifndef EPS
#   define EPS 1.e-6
#endif

/* gpu parameters */
//#define GRID_SIZE 16
//#define BLOCK_SIZE 256



// #define DEBUG

#ifdef DEBUG
#define DPRINTF(fmt, args...) \
do { \
    printf("%s, line %u: " fmt "\r\n", __FUNCTION__, __LINE__ , ##args); \
    fflush(stdout); \
} while (0)
#else   
#define DPRINTF(fmt, args...)   do{}while(0)
#endif


__device__ int get_global_tid()
{
    return (gridDim.x*blockIdx.y + blockIdx.x)*blockDim.x*blockDim.y +
        blockDim.x*threadIdx.y + threadIdx.x;
}

double squared_distance(double* ps, double* center, int dim) {
    double sum = 0;

    for (int i = 0; i < dim; i++){
        double temp = center[i] - ps[i];
        sum += temp * temp;
    }

    return sum;
}

__device__
double squared_distance_on_gpu(double* ps, double* center, int dim) {
    double sum = 0;

    for (int i = 0; i < dim; i++){
        double temp = center[i] - ps[i];
        sum += temp * temp;
    }

    return sum;
}

double** create_2D_double_array_on_gpu(int n, int dim) {
    double **arr;
    arr = (double **)gpu_alloc(n * sizeof(double*));

    for (int i = 0 ; i < n; i++)
        arr[i] = (double *)gpu_alloc( dim * sizeof(double));

    if (arr == NULL ) {
        fprintf(stderr, "Error in allocation!\n");
        exit(-1);
    }

    return arr;
}

double** create_2D_double_array(int n, int dim) {
    double **arr, *temp;
    temp = (double *)calloc(n * dim, sizeof(double));
    arr = (double **)calloc(n, sizeof(double *));

    for (int i = 0 ; i < n; i++)
        arr[i] = temp + i * dim;

    if (arr == NULL || temp == NULL) {
        fprintf(stderr, "Error in allocation!\n");
        exit(-1);
    }

    return arr;
}

void delete_points(double** ps){
    free(ps);
    ps = NULL;
}

double** init_centers_kpp(double **ps, int n, int k, int dim){
    int i;
    int curr_k = 0;
    int first_i;
    int max, max_i;
    double distances_from_centers[n];
    double **centers = create_2D_double_array(k,dim);
    double temp_distances[n];

    // Initialize with max double
    for (i = 0; i < n; i++)
        distances_from_centers[i] = DBL_MAX;

    srand(time(NULL));

    // Choose a first point
    first_i = rand() % n;
    DPRINTF("First random index: %d", first_i);

    memcpy(centers[curr_k], ps[first_i], dim * sizeof(double));
    DPRINTF("Point 1: (%f, %f)", ps[first_i][0], ps[first_i][1]);
    DPRINTF("Center 1: (%f, %f)", centers[curr_k][0], centers[curr_k][1]);

    while(curr_k < k-1) {
        max = -1;
        max_i = -1;
        for(i=0; i<n; i++){
            DPRINTF("New squared_distance: %f and old min squared_distance: %f", squared_distance(ps[i], centers[curr_k], dim), distances_from_centers[i]);
            temp_distances[i] = MIN(squared_distance(ps[i], centers[curr_k], dim), distances_from_centers[i]);  
            if(temp_distances[i] > max){
                max = temp_distances[i];
                max_i = i;
            }
        }
 
        memcpy(distances_from_centers, temp_distances, n * sizeof(double));
        memcpy(centers[++curr_k], ps[max_i], dim * sizeof(double));
    }   
    return centers;
}

int find_cluster_on_cpu(double* ps, double** centers, int n, int k, int dim) {
    int cluster = 0;
    double dist, min = squared_distance(ps, centers[0], dim);

    for (int i = 1; i < k; i++){
        dist = squared_distance(ps, centers[i], dim);
        if (min > dist){
            min = dist;
            cluster = i;
        }
    }

    return cluster;
}

__global__
void find_cluster_on_gpu(double *dev_points, double *dev_centers, int n, int k, int dim, int *result_clusters) {

    double min, dist;
    int index = get_global_tid();

    int start = index*dim;
    int end = start + dim;

    if (index < n){
        for (int i = start; i < end; i+=dim){
            min = DBL_MAX;
            for (int j = 0; j < k; j++){
                dist = squared_distance_on_gpu(&dev_points[i], &dev_centers[j*dim], dim);

                if (min > dist){
                    min = dist;
                    result_clusters[index] = j;
                }
            }
        }
    }
}

void find_clusters_on_gpu(double** points, double** centers, int n, int k, int dim, int* points_clusters,
                          double* dev_points, double* dev_centers, int* map_points_to_clusters, int BLOCK_SIZE) {
    int grid_size = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 gpu_grid(grid_size, 1);
    dim3 gpu_block(BLOCK_SIZE, 1);

    // printf("Grid size : %dx%d\n", gpu_grid.x, gpu_grid.y);
    // printf("Block size: %dx%d\n", gpu_block.x, gpu_block.y);
    // // printf("Shared memory size: %ld bytes\n", shmem_size);

    if (copy_to_gpu(centers[0], dev_centers, k*dim*sizeof(*dev_centers)) != 0) {
        printf("Error in copy_to_gpu centers\n");
        return;
    }

    find_cluster_on_gpu<<<gpu_grid,gpu_block>>>(dev_points, dev_centers, n, k, dim, map_points_to_clusters);

    cudaThreadSynchronize();

    if (copy_from_gpu(points_clusters, map_points_to_clusters, n*sizeof(int)) != 0) {
        printf("Error in copy_to_gpu map_points_to_clusters\n");
        return;
    }
}

double** update_centers(double** ps, int* cls, int n, int k, int dim) {
    int i, j;
    double **new_centers;
    int *points_in_cluster;

    new_centers = create_2D_double_array(k, dim);
    points_in_cluster = (int*) calloc(k, sizeof(int));
 

    for (i = 0; i < n; i++) {
        points_in_cluster[cls[i]]++;
        for (j = 0; j < dim; j++){
            new_centers[cls[i]][j] += ps[i][j];
        }
    }


    for (i = 0; i < k; i++) {
        if (points_in_cluster[i]) {
            for (j = 0; j < dim; j++){
                new_centers[i][j] /= points_in_cluster[i];
            }
        }
    }

    // FIXME: check if points are zero and have no points in cluster
    return new_centers;
}

int main(int argc, char *argv[]) {
    
    int n, k, i, j;
    int dim = 2;
    double **points;
    
    int BLOCK_SIZE = 256; //Default
    if (argc > 1) BLOCK_SIZE = atoi(argv[1]);

    // read input
    scanf("%d %d", &n, &k);
    points = create_2D_double_array(n, dim);
    for (i = 0; i < n; i++) {
        for (j = 0; j < dim; j++)
            scanf("%lf", &points[i][j]);
    }
    
    double **centers;
    centers = init_centers_kpp(points, n, k, dim);

    // start algorithm
    double check = 1;
    double eps = 1.0E-6;
    int *points_clusters;
    double **new_centers;

    points_clusters = (int *)calloc(n, sizeof(int));

    // GPU allocations
    double *dev_centers, *dev_points;
    int *map_points_to_clusters;

    dev_centers = (double *) gpu_alloc(k*dim*sizeof(double));
    dev_points = (double *) gpu_alloc(n*dim*sizeof(double));
    map_points_to_clusters = (int *) gpu_alloc(n*sizeof(int));

    // Copy points to GPU
    if (copy_to_gpu(points[0], dev_points, n*dim*sizeof(*dev_centers)) != 0) {
        printf("Error in copy_to_gpu points\n");
        return -1;
    }

	clock_t start = clock();
	
    while (check > eps) {

        // assign points to clusters - step 1
        find_clusters_on_gpu(points, centers, n, k, dim, points_clusters, dev_points, dev_centers, map_points_to_clusters, BLOCK_SIZE);
        
        // update means - step 2
        new_centers = update_centers(points, points_clusters, n, k, dim);

        // check for convergence
        check = 0;
        for (j = 0; j < k; j++) {
            check += sqrt(squared_distance(new_centers[j], centers[j], dim));
            for (i = 0; i < dim; i++)
                centers[j][i] = new_centers[j][i];
        }
        
        //free new_centers 
        delete_points(new_centers);
    }
    
    double time_elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
	printf("Total Time Elapsed: %lf seconds\n", time_elapsed);
    
    FILE *f;
    //Store Performance metrics
    //For now just the time elapsed, in the future maybe we'll need memory GPU memory bandwidth etc...
    f = fopen("log.out", "w");
    fprintf(f, "Time Elapsed: %lf ", time_elapsed);
    fclose(f);
    
    
    // print & save results
    
    f = fopen("centers.out", "w");
	
	printf("Centers:\n");
    for (i = 0; i < k; i++) {
        for (j = 0; j < dim; j++){
			printf("%lf ", centers[i][j]);
			fprintf(f, "%lf ", centers[i][j]);
		}
        printf("\n");
        fprintf(f, "\n");
    }
    fclose(f);
    
    //Store Mapping Data in case we need it
    copy_from_gpu(points_clusters, map_points_to_clusters, n*sizeof(int));
    f = fopen("point_cluster_map.out", "w");
    for (i =0;i<n;i++){
		fprintf(f, "%d\n", points_clusters[i]);
	}
    
    fclose(f);
    

    // GPU clean
    gpu_free(dev_centers);
    gpu_free(dev_points);
    gpu_free(map_points_to_clusters);

    // clear and exit
    delete_points(points);
    delete_points(centers);
    free(points_clusters);
    return 0;
}
