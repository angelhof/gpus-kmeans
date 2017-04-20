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
#define DIMENSION 4


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

__global__
void find_cluster_on_gpu(double *dev_points, double *dev_centers, int n, int k, int dim, 
                         int *result_clusters) {

    double min, dist;
    int cluster_it_belongs;
    int index = get_global_tid();

    int start = index;
    int end = start + 1;

    if (index < n){
        for (int i = start; i < end; i++){
            min = DBL_MAX;
            for (int j = 0; j < k; j++){
                dist = squared_distance_on_gpu(&dev_points[i*dim], &dev_centers[j*dim], dim);

                if (min > dist){
                    min = dist;
                    cluster_it_belongs = j;
                }
            }
            result_clusters[i] = cluster_it_belongs;
            // printf("points_clusters[%d] = %d\n", i, cluster_it_belongs);
        }
    }
}

// this function is not used
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

    if (index < n){
        for (i = start; i < end; i++) {
            atomicAdd(&dev_points_in_cluster[dev_points_clusters[i]], 1);
            for (j = 0; j < dim; j++) {
                doubleAtomicAdd(&(dev_centers[dev_points_clusters[i]*dim + j]), dev_points[i*dim + j]);
            }
        }
    }
}


__device__
void vectorAddInt(int *dest, int *add, int size) {
    for (int i = 0; i < size; i++) {
        dest[i] += add[i];
    }
}

__device__
void vectorAddDouble(double *dest, double *add, int size) {
    for (int i = 0; i < size; i++) {
        dest[i] += add[i];
    }
}

__global__
void count_points_in_clusters_reduce(double *dev_points,
                                      int *dev_points_clusters,
                                      int n, int k, int dim,
                                      double *dev_new_centers,
                                      int *dev_points_in_cluster) {

    extern __shared__ int arr[];

    int tid = threadIdx.x; // thread id -> [0, block_size)
    int bid = blockIdx.x; // block id -> [0, grid_size)
    int bDim = blockDim.x; // block size
    int gid = bid*bDim + tid; // global id
    int i, j;

    int* s_points_in_cluster = arr; // bDim*k*sizeof(int)
    double* s_new_centers = (double*) (arr + bDim*k); // bDim*k*dim*sizeof(double)

    // initialize shared memory in each block
    // each thread corresponds to one point
    for (i = 0; i < k; i++) {
        if (gid < n && i==dev_points_clusters[gid]) {
            s_points_in_cluster[tid*k + i] = 1;
            for (j = 0; j < dim; j++) {
                s_new_centers[(tid*k + i)*dim + j] = dev_points[gid*dim + j];
            }
        } else {
            s_points_in_cluster[tid*k + i] = 0;
            for (j = 0; j < dim; j++) {
                s_new_centers[(tid*k + i)*dim + j] = 0;
            }
        }
    }
    __syncthreads();


    // the following part is unrolled to improve speed
    // all these if are the reduce process
    if (tid < 128) {
        vectorAddInt(s_points_in_cluster+(tid*k), s_points_in_cluster+((tid+128)*k), k);
        for (i = 0; i < k; i++) {
            vectorAddDouble(s_new_centers+((tid*k + i)*dim), s_new_centers+(((tid+128)*k + i)*dim), dim);
        }
    }
    __syncthreads();

    if (tid < 64) {
        vectorAddInt(s_points_in_cluster+(tid*k), s_points_in_cluster+((tid+64)*k), k);
        for (i = 0; i < k; i++) {
            vectorAddDouble(s_new_centers+((tid*k + i)*dim), s_new_centers+(((tid+64)*k + i)*dim), dim);
        }
    }
    __syncthreads();

    // threads in a warp(=32 threads) are already sychnorized
    if (tid < 32) {
        vectorAddInt(s_points_in_cluster+(tid*k), s_points_in_cluster+((tid+32)*k), k);
        for (i = 0; i < k; i++) {
            vectorAddDouble(s_new_centers+((tid*k + i)*dim), s_new_centers+(((tid+32)*k + i)*dim), dim);
        }
        vectorAddInt(s_points_in_cluster+(tid*k), s_points_in_cluster+((tid+16)*k), k);
        for (i = 0; i < k; i++) {
            vectorAddDouble(s_new_centers+((tid*k + i)*dim), s_new_centers+(((tid+16)*k + i)*dim), dim);
        }
        vectorAddInt(s_points_in_cluster+(tid*k), s_points_in_cluster+((tid+8)*k), k);
        for (i = 0; i < k; i++) {
            vectorAddDouble(s_new_centers+((tid*k + i)*dim), s_new_centers+(((tid+8)*k + i)*dim), dim);
        }
        vectorAddInt(s_points_in_cluster+(tid*k), s_points_in_cluster+((tid+4)*k), k);
        for (i = 0; i < k; i++) {
            vectorAddDouble(s_new_centers+((tid*k + i)*dim), s_new_centers+(((tid+4)*k + i)*dim), dim);
        }
        vectorAddInt(s_points_in_cluster+(tid*k), s_points_in_cluster+((tid+2)*k), k);
        for (i = 0; i < k; i++) {
            vectorAddDouble(s_new_centers+((tid*k + i)*dim), s_new_centers+(((tid+2)*k + i)*dim), dim);
        }
        vectorAddInt(s_points_in_cluster+(tid*k), s_points_in_cluster+((tid+1)*k), k);
        for (i = 0; i < k; i++) {
            vectorAddDouble(s_new_centers+((tid*k + i)*dim), s_new_centers+(((tid+1)*k + i)*dim), dim);
        }
    }

    // each block will have accumulated its result at the start of its shared memory
    // use atomics to add block results
    if (tid == 0) {
        for (i = 0; i < k; i++) {
            atomicAdd(&dev_points_in_cluster[i], s_points_in_cluster[i]);
            for (j = 0; j < dim; j++) {
                doubleAtomicAdd(&(dev_new_centers[i*dim + j]), s_new_centers[i*dim + j]);
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
                // printf("Points in cluster: %d, %d\n", index, dev_points_in_cluster[i]);
            }
        }
    }
}

__device__
void is_converged(double* dev_new_centers, 
                  double* dev_centers, 
                  int* check, 
                  double eps, int index, int dim){
    double diff = sqrt(squared_distance_on_gpu(&dev_new_centers[index], &dev_centers[index], dim));
    if (diff > eps) {
        *check = 0;
    }
}

__global__
void check_convergence(double* dev_centers,
                       double* dev_new_centers,
                       int n, int k, int dim,
                       int* dev_check,
                       double eps){
    int index = get_global_tid();
    if (index < k) {
        *dev_check = 1;
        double diff = squared_distance_on_gpu(&dev_new_centers[index*dim], 
                                              &dev_centers[index*dim], 
                                              dim);
        // printf("Diff[%d]: %lf\n", index, diff);            
        if (diff > eps) {            
            *dev_check = 0;          
        }

        // printf("Center[%d]: (%lf, %lf, %lf)\n", index, dev_centers[index*dim + 0], dev_centers[index*dim + 1], dev_centers[index*dim + 2]);

        for (int i = 0; i < dim; i++){
            // printf("Before Updated dev_centers[%d] = %lf\n", index*dim + i, dev_centers[index*dim + i]);
            dev_centers[index*dim + i] = dev_new_centers[index*dim + i];
            // printf("Updated dev_centers[%d] = %lf\n", index*dim + i, dev_centers[index*dim + i]);
            
        }
    }
}

__global__
void zero_out_arrays(
            double* dev_centers,
            int* dev_points_in_cluster,
            int k, int dim){
    int index = get_global_tid();

    if(index < k*dim){
        dev_centers[index] = 0.0;
    }
    if(index < k){
        dev_points_in_cluster[index] = 0;
    }

}

void kmeans_on_gpu(
            double* dev_points,
            double* dev_centers,
            int n, int k, int dim,
            int* dev_points_clusters,
            int* dev_points_in_cluster,
            double* dev_new_centers,
            int* dev_check,
            int BLOCK_SIZE) {

    
    double eps = 1.0E-4;


    // Calculate grid and block sizes
    int grid_size = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 gpu_grid(grid_size, 1);
    dim3 gpu_block(BLOCK_SIZE, 1);
    
    // printf("Grid size : %dx%d\n", gpu_grid.x, gpu_grid.y);
    // printf("Block size: %dx%d\n", gpu_block.x, gpu_block.y);
    // printf("Shared memory size: %ld bytes\n", shmem_size);

    // Debug
    // printf("%d ", index);

    // assign points to clusters - step 1
    find_cluster_on_gpu<<<gpu_grid, gpu_block>>>(
        dev_points,
        dev_centers,
        n, k, dim,
        dev_points_clusters);
    cudaDeviceSynchronize();
    // printf("index: %d\n", index);
    // if(index < n) printf("dev_points_clusters[%d] = %d\n", index, dev_points_clusters[index]);
    // update means - step 2

    // Clear dev_centers in order to save the new_centers there
    // cudaMemset(dev_points_in_cluster, 0, k*sizeof(int));
    // cudaMemset(dev_centers, 0, k*dim*sizeof(double));
    zero_out_arrays<<<gpu_grid, gpu_block>>>(dev_new_centers, dev_points_in_cluster, k, dim);
    cudaDeviceSynchronize();

    // bDim*k*sizeof(int)
    // bDim*k*dim*sizeof(double)
    // printf("%ld\n", BLOCK_SIZE*k*sizeof(int) + BLOCK_SIZE*k*dim*sizeof(double));

    // this function is new and implemented with partial reduce
    count_points_in_clusters_reduce<<<gpu_grid, gpu_block, BLOCK_SIZE*k*sizeof(int) + BLOCK_SIZE*k*dim*sizeof(double)>>>(
        dev_points, 
        dev_points_clusters, 
        n, k, dim, 
        dev_new_centers, 
        dev_points_in_cluster);
    cudaDeviceSynchronize();


    // Update centers based on counted points
    update_center_on_gpu<<<gpu_grid, gpu_block>>>(
        n, k, dim,
        dev_new_centers,
        dev_points_in_cluster);
    cudaDeviceSynchronize();

    // check for convergence
    check_convergence<<<gpu_grid, gpu_block>>>(
        dev_centers,
        dev_new_centers,
        n, k, dim,
        dev_check,
        eps);
    cudaDeviceSynchronize();
        
}

int main(int argc, char *argv[]) {
    
    int n, k, i, j;
    int dim = 2;
    double **points;
    
    int BLOCK_SIZE = 256; //Default
    if (argc > 1) BLOCK_SIZE = atoi(argv[1]);
    
    //The second input argument should be the dataset filename
    FILE *in;
    if (argc > 2) {
        in = fopen(argv[2], "r");
    } else {
        in = stdin;
    }
    //Parse file
    fscanf(in, "%d %d %d\n", &n ,&k, &dim);
    points = create_2D_double_array(n, dim);
    for (i =0; i<n; i++) {
        for (j=0; j<dim; j++) {
            fscanf(in, "%lf", &points[i][j]);
        }
    }
    fclose(in);
        
    printf("Input Read successfully \n");

    // Calculate grid and block sizes
    int grid_size = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 gpu_grid(grid_size, 1);
    dim3 gpu_block(BLOCK_SIZE, 1);
    
    printf("Grid size : %dx%d\n", gpu_grid.x, gpu_grid.y);
    printf("Block size: %dx%d\n", gpu_block.x, gpu_block.y);
    
    clock_t start = clock();
    
    double **centers;
    printf("Initializing Centers...\n");
    centers = init_centers_kpp(points, n, k, dim);
    printf("Initializing Centers done\n");
    
    // start algorithm
    int *points_clusters;

    points_clusters = (int *)calloc(n, sizeof(int));
    
    // GPU allocations
    double *dev_centers, *dev_points;
    double *dev_new_centers;
    int *dev_points_clusters;
    int *dev_points_in_cluster;

    dev_centers = (double *) gpu_alloc(k*dim*sizeof(double));
    dev_points = (double *) gpu_alloc(n*dim*sizeof(double));
    dev_points_in_cluster = (int *) gpu_alloc(k*sizeof(int));
    dev_points_clusters = (int *) gpu_alloc(n*sizeof(int));
    dev_new_centers = (double *)gpu_alloc(k*dim*sizeof(double));
    
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
    int check = 0;
    int* dev_check = (int *) gpu_alloc(sizeof(int));


    // Debug
    for(i=0;i<k;i++){
        for(j=0;j<dim;j++){
            printf("%lf,\t", centers[i][j]);
        }
        printf("\n");
    }

    while (!check && step < 2000) {
        kmeans_on_gpu(
                dev_points,
                dev_centers,
                n, k, dim,
                dev_points_clusters,
                dev_points_in_cluster,
                dev_new_centers,
                dev_check,
                BLOCK_SIZE);
        

        copy_from_gpu(&check, dev_check, sizeof(int));

        // printf("Step %d\n", step);
        
        step += 1;
    }

    printf("Total num. of steps is %d.\n", step);

    double time_elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;

    printf("Total Time Elapsed: %lf seconds\n", time_elapsed);
    
    printf("Time per step is %lf\n", time_elapsed / step);

    FILE *f;
    //Store Performance metrics
    //For now just the time elapsed, in the future maybe we'll need memory GPU memory bandwidth etc...
    f = fopen("log.out", "w");
    fprintf(f, "Time Elapsed: %lf ", time_elapsed);
    fclose(f);
    
    
    // print & save results
    
    f = fopen("centers.out", "w");
    
    copy_from_gpu(centers[0], dev_centers, k*dim*sizeof(double));
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
