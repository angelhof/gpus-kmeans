#include <float.h>
#include <stdio.h>
#include <cuda.h>
#include "gpu_util.h"
#include "cublas_v2.h"
#include "kmeans_util.h"

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#ifndef EPS
#   define EPS 1.e-6
#endif

#ifdef DEBUG
#define DPRINTF(fmt, args...) \
do { \
    printf("%s, line %u: " fmt "\r\n", __FUNCTION__, __LINE__ , ##args); \
    fflush(stdout); \
} while (0)
#else   
#define DPRINTF(fmt, args...) do{}while(0)
#endif


__constant__ double dev_centers[MAX_CONSTANT_MEMORY];


__device__ int get_global_tid() {
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
double squared_distance_on_gpu(const double* ps, const double* center, const int block_size, const int k, const int dim) {
    double sum = 0;

    for (int i = 0, j=0; i < dim*block_size; i+=block_size,j+=k){
        double temp = center[j] - ps[i];
        // sum = temp * temp + sum as a single operation
        sum = fma(temp, temp, sum);
    }

    return sum;
}

int copy_to_gpu_constant(const double *host, size_t count) {
    double * temp;
    cudaError_t err = cudaGetSymbolAddress((void**)&temp, dev_centers);
    if (err == cudaErrorInvalidValue)
        printf("Error: cudaErrorInvalidValue\n");
    else if (err == cudaErrorInvalidSymbol)
        printf("Error: cudaErrorInvalidSymbol\n");
    else if (err == cudaErrorDuplicateVariableName)
        printf("Error: cudaErrorDuplicateVariableName\n");
    if (copy_to_gpu(host, temp, count) != 0) {
        return -1;
    }
    return 0;
}

int copy_from_gpu_constant(double *host, size_t count) {
    double * temp;
    cudaError_t err = cudaGetSymbolAddress((void**)&temp, dev_centers);
    if (err == cudaErrorInvalidValue)
        printf("Error: cudaErrorInvalidValue\n");
    else if (err == cudaErrorInvalidSymbol)
        printf("Error: cudaErrorInvalidSymbol\n");
    else if (err == cudaErrorDuplicateVariableName)
        printf("Error: cudaErrorDuplicateVariableName\n");
    if (copy_from_gpu(host, temp, count) != 0) {
        return -1;
    }
    return 0;
}

int copy_between_gpu_constant(double *host, size_t count) {
    double * temp;
    cudaError_t err = cudaGetSymbolAddress((void**)&temp, dev_centers);
    if (err == cudaErrorInvalidValue)
        printf("Error: cudaErrorInvalidValue\n");
    else if (err == cudaErrorInvalidSymbol)
        printf("Error: cudaErrorInvalidSymbol\n");
    else if (err == cudaErrorDuplicateVariableName)
        printf("Error: cudaErrorDuplicateVariableName\n");
    if (cudaMemcpy((void*) host, temp, count, cudaMemcpyDeviceToDevice) != cudaSuccess)
        return -1;
    return 0;
}

void transpose(double** src, double* dst, int n, int m){
    int i, j;
    for(i=0; i<n; i++){
        for(j=0; j<m; j++){
            dst[j*n + i] = src[i][j];
        }
    }
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
void find_cluster_on_gpu(const double *dev_points,
                         const int n, const int k, const int dim, 
                         double *result_clusters) {

    double min, dist;
    int cluster_it_belongs = 0;
    register int index = get_global_tid();
    register int j;
    const int thread_id = threadIdx.x;
    extern __shared__ double local_points[];
    int block_size = blockDim.x;

    if (index < n){
        for (j = 0; j < dim; ++j){
            local_points[thread_id + j*block_size] = dev_points[index + j];
            // if (index < k)
            //     printf("dev_centers[%d][%d] = %lf\n", index, j, dev_centers[j * k + index]);
        }

        min = DBL_MAX;
        for (j = 0; j < k; ++j){
            result_clusters[j*n + index] = 0.0;
            dist = squared_distance_on_gpu(&local_points[thread_id], &dev_centers[j], block_size, k, dim);
            if (min > dist){
                min = dist;
                cluster_it_belongs = j;
            }
        }
        result_clusters[cluster_it_belongs*n + index] = 1.0;
        for (int j = 0; j < k; j++){
            printf("result_clusters[%d][%d] = %lf --> line[%d]\n", j, index, result_clusters[j*n + index], index+2);
        }
    }
}

__global__
void update_center_on_gpu(const int k, const int dim, 
              double* dev_new_centers,
                          const double* dev_points_in_cluster){
    int j;
    const int index = get_global_tid();

    // do all numbers in k*dim threads 
    if (index < k){
        if (dev_points_in_cluster[index] > 0) {
            #pragma unroll
            for (j = 0; j < dim; j++){
                printf("Before: Dev_new_centers[%d][%d] = %d\n", index, j, dev_new_centers[j*k+index]);
                dev_new_centers[j*k + index] /= dev_points_in_cluster[index];
                printf("After: Dev_new_centers[%d][%d] = %d\n", index, j, dev_new_centers[j*k+index]);
            }
        }
    }
}

__global__
void create_dev_ones(double* dev_ones, int n) {
    int index = get_global_tid();

    if(index < n){
        dev_ones[index] = 1.0;
    }
}

// Just a wrapper function of create_dev_ones to avoid putting that
// function into kmeans_gpu. (create_dev_ones is used in main)
void call_create_dev_ones(double* dev_ones, int n, dim3 gpu_grid, dim3 gpu_block) {
    create_dev_ones<<<gpu_grid,gpu_block>>>(dev_ones, n);
    cudaDeviceSynchronize();
}


void swap(double** src, double** dst){
    double *temp = *src;
    *src = *dst;
    *dst = temp;
}

void kmeans_on_gpu(
            double* dev_points,
            int n, int k, int dim,
            double* dev_points_clusters,
            double* dev_points_in_cluster,
            double* dev_new_centers,
            int* dev_check,
            int BLOCK_SIZE, 
            cublasHandle_t handle,
            cublasStatus_t stat,
            double* dev_ones,
            double* dev_temp_centers) {

    double alpha = 1.0, beta = 0.0;

    // Calculate grid and block sizes
    int grid_size = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 gpu_grid(grid_size, 1);
    dim3 gpu_block(BLOCK_SIZE, 1);
    
    // printf("Grid size : %dx%d\n", gpu_grid.x, gpu_grid.y);
    // printf("Block size: %dx%d\n", gpu_block.x, gpu_block.y);
    // printf("Shared memory size: %ld bytes\n", shmem_size);
    printf("A\n");
    // assign points to clusters - step 1
    find_cluster_on_gpu<<<gpu_grid,gpu_block, BLOCK_SIZE*dim*sizeof(double)>>>(
        dev_points,
        n, k, dim,
        dev_points_clusters);
    cudaDeviceSynchronize();
    printf("B\n");
    // update means - step 2
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                k, dim, n,
                &alpha,
                dev_points_clusters, n,
                dev_points, n,
                &beta,
                dev_new_centers, k);
    // cudaDeviceSynchronize();
    printf("C\n");
    cublasDgemv(handle, CUBLAS_OP_T,
                n, k,
                &alpha,
                dev_points_clusters, n,
                dev_ones, 1,
                &beta,
                dev_points_in_cluster, 1);
    // cudaDeviceSynchronize();
    printf("D\n");
    // Update centers based on counted points
    update_center_on_gpu<<<gpu_grid,gpu_block>>>(
        k, dim,
        dev_new_centers,
        dev_points_in_cluster);
    cudaDeviceSynchronize();
    printf("E\n");
    //Check for convergence with CUBLAS
    //dev_new_centers and dev_centers arrays are actually checked for equality
    //No distances are calculated separately for each center point.
    //It seems like its working smoothly so far
    int icheck = 0; //This is used to make it compatible with how the code works now
    double check = 0.0;
    //First subtract the dev_center arrays
    alpha = -1.0;

    cublasDaxpy(handle, k*dim, &alpha, dev_new_centers, 1, dev_centers, 1);
    // cudaDeviceSynchronize();
    //Now find the norm2 of the new_centers
    // cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST);
    cublasDnrm2(handle, k*dim, dev_centers, 1, &check);
    if (!(check > EPS)) icheck = 1;
    copy_to_gpu(&icheck, dev_check, sizeof(int));
    printf("E\n");
    //Update new centers
    // TODO: Swap pointers
    cudaDeviceSynchronize();
    if(copy_between_gpu_constant(dev_new_centers, k * dim * sizeof(double)) != 0){
        printf("Error in copy_between_gpu_constant centers\n");
    }
    cudaDeviceSynchronize();
    
    printf("F\n");
}