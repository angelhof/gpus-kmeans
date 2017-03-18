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
#define GRID_SIZE 16
#define BLOCK_SIZE 256

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


double squared_distance(double* ps, double* center, int dim) {
    double sum = 0;

    for (int i = 0; i < dim; i++){
        double temp = center[i] - ps[i];
        sum += temp * temp;
    }

    return sum;
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


__global__
int find_cluster(double** points, double** centers, int n, int k, int dim) {
    int cluster = 0, tid =  blockIdx.x;

    double dist, min = squared_distance(points[tid], centers[0], dim);
    for (int i = 1; i < k; i++){
        dist = squared_distance(points[tid], centers[i], dim);
        if (min > dist){
            min = dist;
            cluster = i;
        }
    }

    return cluster;
}

void find_clusters(double** points, double** centers, int n, int k, int dim, int* points_clusters) {
	for (int i = 0; i < n; i++) {
        points_clusters[i] = find_cluster(points[i], centers, n, k, dim);
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

int main() {
    
    int n, k, i, j;
    int dim = 2;
    double **points;

    // read input
    scanf("%d %d", &n, &k);
    points = create_2D_double_array(n, dim);
    for (i = 0; i < n; i++)
    	for (j = 0; j < dim; j++)
    		scanf("%lf", &points[i][j]);

    
    dim3 gpu_grid(GRID_SIZE, 1);
    dim3 gpu_block(BLOCK_SIZE, 1);
    // size_t shmem_size = block_size * sizeof(float);

    printf("Grid size : %dx%d\n", gpu_grid.x, gpu_grid.y);
    printf("Block size: %dx%d\n", gpu_block.x, gpu_block.y);
    // printf("Shared memory size: %ld bytes\n", shmem_size);

    // GPU allocations
    value_t *gpu_A = (value_t *)gpu_alloc(n*n*sizeof(*gpu_A));
    if (!gpu_A) error(0, "gpu_alloc failed: %s", gpu_get_last_errmsg());
    
    
    double **centers;
    centers = init_centers_kpp(points, n, k, dim);

    // start algorithm
    double check = 1;
    double eps = 1.0E-6;
    int *points_clusters;
    double **new_centers;
    new_centers = create_2D_double_array(k, dim);
    points_clusters = (int *)calloc(n, sizeof(int));

    while (check > eps) {

        // assign points to clusters - step 1
        find_clusters(points, centers, n, k, dim, points_clusters);
        
        // update means - step 2
        new_centers = update_centers(points, points_clusters, n, k, dim);

        // check for convergence
        check = 0;
        for (j = 0; j < k; j++) {
            check += sqrt(squared_distance(new_centers[j], centers[j], dim));
            for (i = 0; i < dim; i++)
            	centers[j][i] = new_centers[j][i];
        }
    }

    // print results
    printf("Centers:\n");
    for (i = 0; i < k; i++) {
        for (j = 0; j < dim; j++)
            printf("%lf ", centers[i][j]);
        printf("\n");
    }

    // clear and exit
    delete_points(points);
    delete_points(centers);
    return 0;
}