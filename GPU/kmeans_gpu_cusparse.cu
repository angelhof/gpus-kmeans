#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gpu_util.h"
#include "kmeans_util_cusparse.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"


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

int main(int argc, char *argv[]) {
    
    int n, k, i, j;
    int dim = 2;
    double **points;
    
    int block_size = 256;
    int grid_size = 1024;
    if (argc > 1) block_size = atoi(argv[1]);
    if (argc > 2) grid_size = atoi(argv[2]);
    
    //The second input argument should be the dataset filename
    FILE *in;
    if (argc > 3) {
        in = fopen(argv[3], "r");
    } else {
        in = stdin;
    }
    //Parse file
    register short read_items = -1;
    read_items = fscanf(in, "%d %d %d\n", &n ,&k, &dim);
    if (read_items != 3){
        printf("Something went wrong with reading the parameters!\n");
        return EXIT_FAILURE;
    }
    points = create_2D_double_array(n, dim);
    for (i =0; i<n; i++) {
        for (j=0; j<dim; j++) {
            read_items = fscanf(in, "%lf", &points[i][j]);
            if (read_items != 1) {
                printf("Something went wrong with reading the points!\n");
            }
        }
    }
    fclose(in);
        
    printf("Input Read successfully \n");
    
    //Create CUBLAS Handles
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed!\n");
        return EXIT_FAILURE;
    }

    cusparseHandle_t cusparse_handle;
    cusparseStatus_t cusparse_stat;

    cusparse_stat = cusparseCreate(&cusparse_handle);
    if (cusparse_stat != CUSPARSE_STATUS_SUCCESS) {
        printf ("CUSPARSE initialization failed!\n");
        return EXIT_FAILURE;
    }
    
    // Calculate grid and block sizes
    dim3 gpu_grid((n+block_size-1)/block_size, 1);
    dim3 gpu_block(block_size, 1);
    
    printf("Grid size : %dx%d\n", grid_size, 1);
    printf("Block size: %dx%d\n", block_size, 1);
    
    clock_t start = clock();
    
    double **centers;
    printf("Initializing Centers...\n");
    centers = init_centers_kpp(points, n, k, dim);
    printf("Initializing Centers done\n");
    
    // start algorithm
    
    // GPU allocations
    double *dev_centers, *dev_points;
    double *dev_new_centers;
    // double *dev_points_clusters;
    int *dev_points_in_cluster;
    double *dev_ones;
    // int* dev_nnzPerRow;
    double* dev_csrVal_points_clusters;
    int* dev_csrRowPtr_points_clsusters;
    int* dev_csrColInd_points_clsusters;

    dev_centers = (double *) gpu_alloc(k*dim*sizeof(double));
    dev_points = (double *) gpu_alloc(n*dim*sizeof(double));
    dev_points_in_cluster = (int *) gpu_alloc(k*sizeof(int));
    // dev_points_clusters = (double *) gpu_alloc(n*k*sizeof(double));
    dev_new_centers = (double *) gpu_alloc(k*dim*sizeof(double));
    dev_ones = (double *) gpu_alloc(n*sizeof(double));
    // dev_nnzPerRow = (int *) gpu_alloc(k*sizeof(int));
    dev_csrVal_points_clusters = (double *) gpu_alloc(n*sizeof(double));
    dev_csrRowPtr_points_clsusters = (int *) gpu_alloc((k+1)*sizeof(int));
    dev_csrColInd_points_clsusters = (int *) gpu_alloc(n*sizeof(int));
    
    printf("GPU allocs done \n");
    
    call_create_dev_ones(dev_ones, n, gpu_grid, gpu_block);

    // Transpose points and centers for cublas
    // TODO: Transpose at cublas in gpu
    double * staging_points = (double*) calloc(n*dim, sizeof(double));
    double * staging_centers = (double*) calloc(k*dim, sizeof(double));
    transpose(points, staging_points, n, dim);
    transpose(centers, staging_centers, k, dim);
    
    // Synchronize is for dev_ones
    cudaDeviceSynchronize();

    // Copy points to GPU
    if (copy_to_gpu(staging_points, dev_points, n*dim*sizeof(double)) != 0) {
        printf("Error in copy_to_gpu points\n");
        return -1;
    }

    // Copy centers to GPU
    if (copy_to_gpu(staging_centers, dev_centers, k*dim*sizeof(double)) != 0) {
        printf("Error in copy_to_gpu centers\n");
        return -1;
    }

    // Debug
    // printf("Initial centers:\n");
    // for(i=0;i<k;i++){
    //     for(j=0;j<dim;j++)
    //         printf("%lf,\t", centers[i][j]);
    //     printf("\n");
    // }
    
    int step = 0;
    int check = 0;

    printf("Loop Start...\n");
    while (!check) {
        check = kmeans_on_gpu(
                    dev_points,
                    dev_centers,
                    n, k, dim,
                    dev_points_in_cluster,
                    dev_new_centers,
                    block_size,
                    grid_size,
                    handle,
                    dev_ones,
                    cusparse_handle,
                    dev_csrVal_points_clusters,
                    dev_csrRowPtr_points_clsusters,
                    dev_csrColInd_points_clsusters);
        
        // printf("Step %d Check: %d \n", step, check);
        
        step += 1;
        // if (step == 3) break;
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
    
    copy_from_gpu(staging_centers, dev_centers, k*dim*sizeof(double));
    printf("Centers:\n");
    for (i = 0; i < k; i++) {
        for (j = 0; j < dim; j++){
            printf("%lf,\t", staging_centers[j*k + i]);
            fprintf(f, "%lf ", staging_centers[j*k + i]);
        }
        printf("\n");
        fprintf(f, "\n");
    }
    fclose(f);
    
    //Store Mapping Data in case we need it
    // int *points_clusters;
    // points_clusters = (int *)calloc(n, sizeof(int));
    // copy_from_gpu(points_clusters, dev_points_clusters, n*sizeof(int));
    // f = fopen("point_cluster_map.out", "w");
    // for (i =0;i<n;i++){
    //     fprintf(f, "%d\n", points_clusters[i]);
    // }
    
    // fclose(f);
    
    // GPU clean
    gpu_free(dev_centers);
    gpu_free(dev_points);
    gpu_free(dev_points_in_cluster);
    // gpu_free(dev_points_clusters);
    gpu_free(dev_new_centers);
    gpu_free(dev_ones);
    // gpu_free(dev_nnzPerRow);
    gpu_free(dev_csrVal_points_clusters);
    gpu_free(dev_csrRowPtr_points_clsusters);
    gpu_free(dev_csrColInd_points_clsusters);

    stat = cublasDestroy(handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed!\n");
        return EXIT_FAILURE;
    }

    cusparse_stat = cusparseDestroy(cusparse_handle);
    if (cusparse_stat != CUSPARSE_STATUS_SUCCESS) {
        printf ("CUSPARSE initialization failed!\n");
        return EXIT_FAILURE;
    }

    // clear and exit
    delete_points(points);
    delete_points(centers);
    // free(points_clusters);
    return 0;
}
