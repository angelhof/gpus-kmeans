#include <float.h>
#include <stdio.h>
#include <cuda.h>
#include "gpu_util.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#ifndef EPS
#   define EPS 1.e-6
#endif

// #define DEBUG

#ifdef DEBUG
#define DPRINTF(fmt, args...) \
do { \
    printf("%s, line %u: " fmt "\r\n", __FUNCTION__, __LINE__ , ##args); \
    fflush(stdout); \
} while (0)
#else   
#define DPRINTF(fmt, args...) do{}while(0)
#endif

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
double squared_distance_on_gpu(const double* ps, const double* center, const int n, const int k, const int dim) {
    double sum = 0;

    for (int i = 0, j=0; i < dim*n; i+=n,j+=k){
        double temp = center[j] - ps[i];
        sum += temp * temp;
    }

    return sum;
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
void find_cluster_on_gpu(const double *dev_points, const double *dev_centers, 
                         const int n, const int k, const int dim,
                         const int step,
                         // double *result_clusters,
                         int *cols) {

    double min, dist;
    int cluster_it_belongs;
    const unsigned int index = get_global_tid();

    if (index < n){
        for (int i = index; i < n; i += step) {
            min = DBL_MAX;
            for (int j = 0; j < k; j++){
                // result_clusters[j*n + index] = 0.0;
                dist = squared_distance_on_gpu(&dev_points[i], &dev_centers[j], n, k, dim);

                if (min > dist){
                    min = dist;
                    cluster_it_belongs = j;
                }
                // min = fmin(min, dist);
                // cluster_it_belongs = cluster_it_belongs ^ ((j ^ cluster_it_belongs) & (min == dist));
            }
            // Only 1 in the cluster it belongs and everything else 0
            // result_clusters[cluster_it_belongs*n + i] = 1.0;
            cols[i] = cluster_it_belongs;
            // for (int j = 0; j < k; j++){
            //     printf("result_clusters[%d][%d] = %lf --> line[%d]\n", j, i, result_clusters[j*n + i], i+2);
            // }
        }
    }
}

__global__
void update_center_on_gpu(const int size, const int k,
                          // const int n, const int k, const int dim, 
                          double* dev_centers, 
                          const int* dev_points_in_cluster){

    // const unsigned int j = blockIdx.x;
    // const unsigned int i = threadIdx.x;
    const unsigned int index = get_global_tid();
    const unsigned int cluster = index % k;
    // printf("i = %d, j = %d, index = %d, k*dim = %d\n", i, j, j*k + i, k*dim);
    // if (j < dim && i < k){
    if (index < size){
        // printf("dev_points_in_cluster[%d] = %d\n", i, (int)dev_points_in_cluster[i]);
        // printf("dev_centers[%d][%d] = %lf\n", i, j, dev_centers[i*dim + j]);
        if (dev_points_in_cluster[cluster] > 0) {
            // for (int j = 0; j < dim; j++){
                dev_centers[index] /= dev_points_in_cluster[cluster];
            // }
            // printf("Points in cluster: %d, %d\n", j*k + i, dev_points_in_cluster[i]);
        }
        // printf("new_dev_centers[%d][%d] = %lf\n", i, j, dev_centers[i*dim + j]);
    }
}

__global__
void create_dev_ones(double* dev_ones, int n) {
    int index = get_global_tid();

    if(index < n){
        dev_ones[index] = 1.0;
    }
}

void return_dense_dev_point_to_cluster_map(
    double* dev_point_to_cluster_map,
    const int k, 
    const int n,
    cusparseHandle_t cusparse_handle,
    double* dev_ones,
    int* dev_csrRowPtr_points_clsusters,
    int* dev_csrColInd_points_clsusters
    )
{
    cusparseMatDescr_t descrA;
    cusparseStatus_t cusparseStatus;

    // Initialization before making the sparse matrix back to dense
    cusparseStatus = cusparseCreateMatDescr(&descrA);
    if(cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        printf("cusparseCreateMatDescr returned error code %d!\n", cusparseStatus);
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
    cusparseDcsr2dense(cusparse_handle, 
                       k, n, 
                       descrA, 
                       dev_ones,
                       dev_csrRowPtr_points_clsusters,
                       dev_csrColInd_points_clsusters, 
                       dev_point_to_cluster_map, 
                       k);
    cusparseDestroyMatDescr(descrA);
}
// Just a wrapper function of create_dev_ones to avoid putting that
// function into kmeans_gpu. (create_dev_ones is used in main)
void call_create_dev_ones(double* dev_ones, int n, dim3 gpu_grid, dim3 gpu_block) {
    create_dev_ones<<<gpu_grid,gpu_block>>>(dev_ones, n);
}

int kmeans_on_gpu(
            const double* dev_points,
            double* dev_centers,
            const int n, const int k, const int dim,
            int* dev_points_in_cluster,
            double* dev_new_centers,
            const int block_size,
            const int grid_size,
            //CUBLAS shit
            cublasHandle_t handle,
            const double* dev_ones,
            //CUSPARSE shit
            cusparseHandle_t cusparse_handle,
            double* dev_csrVal_points_clusters,
            int* dev_csrRowPtr_points_clsusters,
            int* dev_csrColInd_points_clsusters) {

    double alpha = 1.0, beta = 0.0;

    dim3 gpu_grid(grid_size, 1);
    dim3 gpu_block(block_size, 1);
    
    // printf("Grid size : %dx%d\n", gpu_grid.x, gpu_grid.y);
    // printf("Block size: %dx%d\n", gpu_block.x, gpu_block.y);
    // printf("Shared memory size: %ld bytes\n", shmem_size);

    // assign points to clusters - step 1
    const int step = block_size * grid_size;
    find_cluster_on_gpu<<<gpu_grid, gpu_block>>>(
        dev_points,
        dev_centers,
        n, k, dim,
        step,
        // dev_points_clusters,
        dev_csrColInd_points_clsusters);
    cudaDeviceSynchronize();

    int * temp = (int *) malloc(n*sizeof(int));
    copy_from_gpu(temp, dev_csrColInd_points_clsusters, n*sizeof(int));
    
    int * cols;
    cols = (int*) malloc(n*sizeof(int));
    int * rows;
    rows = (int *) malloc((k+1)*sizeof(int));
    int * host_points_in_cluster;
    host_points_in_cluster = (int *) calloc(k, sizeof(int));
    int ii = 0;
    for (int i = 0; i < k; i++) {
        char first = 0;
        for(int j = 0; j < n; j++) {
            if (temp[j] == i) {
                cols[ii++] = j;
                host_points_in_cluster[i]++;
                if (!first) {
                    rows[i] = ii - 1;
                    first = 1;
                }
            }
        }
    }
    rows[k] = n;
    copy_to_gpu(rows, dev_csrRowPtr_points_clsusters, (k+1)*sizeof(int));
    copy_to_gpu(cols, dev_csrColInd_points_clsusters, n*sizeof(int));
    copy_to_gpu(host_points_in_cluster, dev_points_in_cluster, k*sizeof(int));
    free(temp);
    free(cols);
    free(host_points_in_cluster);
    free(rows);

    // update means - step 2
    cusparseMatDescr_t descrA;
    cusparseStatus_t cusparseStatus;

    cusparseStatus = cusparseCreateMatDescr(&descrA);
    if(cusparseStatus != CUSPARSE_STATUS_SUCCESS)
        printf("cusparseCreateMatDescr returned error code %d!\n", cusparseStatus);

    // cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
    // cusparseSetMatFillMode(descrA,CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
    // cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);

    // ATTENTION: Check if can be avoided!!
    // double * dev_points_clusters_t;
    // dev_points_clusters_t = (double *) gpu_alloc(k*n*sizeof(double));
    // cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
    //             k, n, &alpha,
    //             dev_points_clusters, n,
    //             &beta,
    //             dev_points_clusters, n,
    //             dev_points_clusters_t, k);
    // cudaDeviceSynchronize();

    // cusparseStatus = cusparseDdense2csr(cusparse_handle, k, n, descrA,
    //                    dev_points_clusters_t,
    //                    k, nnnn,
    //                    dev_csrVal_points_clusters,
    //                    dev_csrRowPtr_points_clsusters,
    //                    dev_csrColInd_points_clsusters);
    // if(cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    //     printf("cusparseDdense2csr returned error code %d!\n", cusparseStatus);

   cusparseDcsrmm(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   k, dim, n, n,
                   &alpha,
                   descrA,
                   dev_ones,
                   dev_csrRowPtr_points_clsusters,
                   dev_csrColInd_points_clsusters,
                   dev_points, n,
                   &beta,
                   dev_new_centers, k);
    // cudaDeviceSynchronize();

    


    const int size = k * dim;
    // Update centers based on counted points
    update_center_on_gpu<<<gpu_grid, gpu_block>>>(
        size, k,
        dev_new_centers,
        dev_points_in_cluster);
    cudaDeviceSynchronize();
    //Check for convergence with CUBLAS
    //dev_new_centers and dev_centers arrays are actually checked for equality
    //No distances are calculated separately for each center point.
    //It seems like its working smoothly so far
    int icheck = 0;
    double check = 0.0;
    //First subtract the dev_center arrays
    alpha = -1.0;
    cublasDaxpy(handle, k*dim, &alpha, dev_new_centers, 1, dev_centers, 1);

    // Now find the norm2 of the new_centers
    cublasDnrm2(handle, k*dim, dev_centers, 1, &check);

    //Update new centers
    // TODO: Swap pointers
    cudaMemcpy(dev_centers, dev_new_centers, sizeof(double)*k*dim, cudaMemcpyDeviceToDevice);
    
    if (check < EPS) icheck = 1;
    return icheck;
}
