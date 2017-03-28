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



//#define DEBUG

#ifdef DEBUG
#define DPRINTF(fmt, args...) \
do { \
    printf("%s, line %u: " fmt "\r\n", __FUNCTION__, __LINE__ , ##args); \
    fflush(stdout); \
} while (0)
#else   
#define DPRINTF(fmt, args...)   do{}while(0)
#endif


#if __CUDA_ARCH__ < 600
__device__ double doubleAtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
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
    double *distances_from_centers, *temp_distances;
    distances_from_centers = (double*) malloc(sizeof(double)*n);
    double **centers = create_2D_double_array(k,dim);
    temp_distances = (double*) malloc(sizeof(double)*n);
    
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
    
    free(temp_distances);
    free(distances_from_centers);
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
void find_cluster_on_gpu(double *dev_points, double *dev_centers, int n, int k, int dim, 
                         int *result_clusters) {

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

// void find_clusters_on_gpu(int n, int k, int dim, double* dev_points, double* dev_centers, int* dev_points_clusters, int BLOCK_SIZE) {
//     int grid_size = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
//     dim3 gpu_grid(grid_size, 1);
//     dim3 gpu_block(BLOCK_SIZE, 1);
    
//     // printf("Grid size : %dx%d\n", gpu_grid.x, gpu_grid.y);
//     // printf("Block size: %dx%d\n", gpu_block.x, gpu_block.y);
//     // // printf("Shared memory size: %ld bytes\n", shmem_size);

//     // centers already copied in GPU
//     find_cluster_on_gpu<<<gpu_grid,gpu_block>>>(dev_points, dev_centers, n, k, dim, dev_points_clusters);
    
//     cudaDeviceSynchronize();

//     // next function will read from dev_points_clusters
// }

__global__
void count_points_in_clusters_on_gpu(double* dev_points,       // Device point data 
                                     int* dev_points_clusters, // Device point -> cluster
                                     int n, int k, int dim,  
                                     double* dev_centers,      // Device center data    
                                     int* dev_points_in_cluster) {
    int i, j;
    
    int index = get_global_tid();

    int start = index;
    int end = start + 1;

    // Clear dev_centers in order to save the new_centers there
    if (index < k){
        dev_points_in_cluster[index] = 0;
        for(j=0; j<dim; j++){
            dev_centers[index*dim + j] = 0;
        }
    }
    __syncthreads();


    
    if (index < n){
        for (i = start; i < end; i++) {
            atomicAdd(&dev_points_in_cluster[dev_points_clusters[i]], 1);
            for (j = 0; j < dim; j++) {
                doubleAtomicAdd(&(dev_centers[dev_points_clusters[i]*dim + j]), dev_points[i*dim + j]);
            }
        }
    }

    
}

__global__
void update_center_on_gpu(int n, int k, int dim, 
                          double* dev_centers, 
                          int* dev_points_in_cluster){
    int i, j;

    int index = get_global_tid();

    int start = index;
    int end = start + 1;

    if (index < k){
        for (i = start; i < end; i++) {
            if (dev_points_in_cluster[i]) {
                for (j = 0; j < dim; j++){
                    dev_centers[i*dim + j] /= dev_points_in_cluster[i];
                }
            }
        }
    }
}

// void update_centers_on_gpu(int n, int k, int dim,
//                            double* dev_points,        // Device point data 
//                            double* dev_centers,       // Device old center data
//                            int* dev_points_clusters,  // Device points -> clusters
//                            int BLOCK_SIZE,              
//                            int* dev_points_in_cluster) {  // How many points in each cluster
//     int grid_size = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
//     dim3 gpu_grid(grid_size, 1);
//     dim3 gpu_block(BLOCK_SIZE, 1);
    
//     // printf("Grid size : %dx%d\n", gpu_grid.x, gpu_grid.y);
//     // printf("Block size: %dx%d\n", gpu_block.x, gpu_block.y);
//     // // printf("Shared memory size: %ld bytes\n", shmem_size);
    

//     // dev_points_clusters are in GPU from previous function


//     // Count points that belong to each cluster
//     count_points_in_clusters_on_gpu<<<gpu_grid,gpu_block>>>(
//         dev_points, 
//         dev_points_clusters, 
//         n, k, dim, 
//         dev_centers, 
//         dev_points_in_cluster);

//     cudaDeviceSynchronize();

//     // Update centers based on counted points
//     update_center_on_gpu<<<gpu_grid,gpu_block>>>(n, k, dim, dev_centers, dev_points_in_cluster);

//     cudaDeviceSynchronize();

//     // new_centers are copied to CPU in while
// }

int main(int argc, char *argv[]) {
    
    int n, k, i, j;
    int dim = 2;
    double **points;
    
    int BLOCK_SIZE = 256; //Default
    if (argc > 1) BLOCK_SIZE = atoi(argv[1]);
    
    //The second input argument should be the dataset filename
    if (argc > 2) {
        FILE *in;
        in = fopen(argv[2], "r");
        //Parse file
        fscanf(in, "%d %d \n", &n ,&k);
        points = create_2D_double_array(n, dim);
        for (i =0; i<n; i++) {
            for (j=0; j<dim; j++) {
                fscanf(in, "%lf", &points[i][j]);
            }
        }
        fclose(in);
    //Otherwise parse stdin
    //PS: For large datasets this doesn't work at all
    } else {
        // read input
        scanf("%d %d", &n, &k);
        points = create_2D_double_array(n, dim);
        for (i = 0; i < n; i++) {
            for (j = 0; j < dim; j++)
                scanf("%lf", &points[i][j]);
        }
    }
        
    printf("Input Read successfully \n");

    // Calculate grid and block sizes
    int grid_size = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 gpu_grid(grid_size, 1);
    dim3 gpu_block(BLOCK_SIZE, 1);
    
    printf("Grid size : %dx%d\n", gpu_grid.x, gpu_grid.y);
    printf("Block size: %dx%d\n", gpu_block.x, gpu_block.y);
    // printf("Shared memory size: %ld bytes\n", shmem_size);
    
    clock_t start = clock();
    
    double **centers;
    printf("Initializing Centers...\n");
    centers = init_centers_kpp(points, n, k, dim);
    printf("Initializing Centers done\n");
    
    // start algorithm
    double check = 1;
    double eps = 1.0E-6;
    int *points_clusters;
    double **new_centers;

    new_centers = create_2D_double_array(k, dim);
    points_clusters = (int *)calloc(n, sizeof(int));
    
    // GPU allocations
    double *dev_centers, *dev_points;
    int *dev_points_clusters;
    int *dev_points_in_cluster;

    dev_centers = (double *) gpu_alloc(k*dim*sizeof(double));
    dev_points = (double *) gpu_alloc(n*dim*sizeof(double));
    dev_points_in_cluster = (int *) gpu_alloc(k*sizeof(int));
    dev_points_clusters = (int *) gpu_alloc(n*sizeof(int));
    
    printf("GPU allocs done \n");
    
    // Copy points to GPU
    if (copy_to_gpu(points[0], dev_points, n*dim*sizeof(double)) != 0) {
        printf("Error in copy_to_gpu points\n");
        return -1;
    }

    // Copy centers to GPU
    if (copy_to_gpu(centers[0], dev_centers, k*dim*sizeof(double)) != 0) {
        printf("Error in copy_to_gpu centers\n");
        return -1;
    }
    
    printf("Loop Start \n");
    int step = 0;
    while (check > eps) {

        // assign points to clusters - step 1
        // find_clusters_on_gpu(n, k, dim, dev_points, dev_centers, dev_points_clusters, BLOCK_SIZE);
        find_cluster_on_gpu<<<gpu_grid,gpu_block>>>(dev_points, dev_centers, n, k, dim, dev_points_clusters);
        cudaDeviceSynchronize();
        
        // update means - step 2
        // update_centers_on_gpu(n, k, dim,
        //     dev_points,
        //     dev_centers,
        //     dev_points_clusters,
        //     BLOCK_SIZE,
        //     dev_points_in_cluster);

        // Count points that belong to each cluster
        count_points_in_clusters_on_gpu<<<gpu_grid,gpu_block>>>(
            dev_points, 
            dev_points_clusters, 
            n, k, dim, 
            dev_centers, 
            dev_points_in_cluster);
        cudaDeviceSynchronize();

        // Update centers based on counted points
        update_center_on_gpu<<<gpu_grid,gpu_block>>>(n, k, dim, dev_centers, dev_points_in_cluster);
        cudaDeviceSynchronize();

        // TODO: centers check in GPU, so we don't copy from gpu each time
        if (copy_from_gpu(new_centers[0], dev_centers, k*dim*sizeof(double)) != 0) {
            printf("Error in copy_from_gpu dev_centers\n");
            return -1;
        }

        // check for convergence
        for (i = 0; i < k; i++){
            for (j = 0; j < dim; j++){
                printf("%lf ", new_centers[i][j]);
            }
            printf("\n");
        }

        check = 0;
        for (j = 0; j < k; j++) {
            check += sqrt(squared_distance(new_centers[j], centers[j], dim));
            for (i = 0; i < dim; i++){
                centers[j][i] = new_centers[j][i];
            }
        }
        
        printf("Step %d , Convergence: %lf \n", step, check);
        step += 1;
        //free new_centers
        // if (step == 5) break;
        // delete_points(new_centers);
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
    copy_from_gpu(points_clusters, dev_points_clusters, n*sizeof(int));
    f = fopen("point_cluster_map.out", "w");
    for (i =0;i<n;i++){
        fprintf(f, "%d\n", points_clusters[i]);
    }
    
    fclose(f);
    
    // GPU clean
    gpu_free(dev_centers);
    gpu_free(dev_points);
    gpu_free(dev_points_clusters);

    // clear and exit
    delete_points(points);
    delete_points(centers);
    free(points_clusters);
    return 0;
}
