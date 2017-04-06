#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gpu_util.h"
#include "kmeans_util.h"
#include "cublas_v2.h"

/* gpu parameters */
//#define GRID_SIZE 16
//#define BLOCK_SIZE 256
#define DIMENSION 2


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
    
    //Create CUBLAS Handles
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    
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
    double *dev_points_clusters;
    double *dev_points_in_cluster;
    double *dev_ones;

    dev_centers = (double *) gpu_alloc(k*dim*sizeof(double));
    dev_points = (double *) gpu_alloc(n*dim*sizeof(double));
    dev_points_in_cluster = (double *) gpu_alloc(k*sizeof(double));
    dev_points_clusters = (double *) gpu_alloc(n*k*sizeof(double));
    dev_new_centers = (double *) gpu_alloc(k*dim*sizeof(double));
    dev_ones = (double *) gpu_alloc(n*sizeof(double));
    
    printf("GPU allocs done \n");
    
    call_create_dev_ones(dev_ones, n, gpu_grid, gpu_block);

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

    // FIXME: For now we pass TWO matrices for centers, one normal and 
    //        one transposed. The transposed can be omitted by doing some
    //        changes in Step 1 of K-Means.
    double *dev_temp_centers;
    dev_temp_centers = (double *) gpu_alloc(k*dim*sizeof(double));

    printf("Loop Start \n");
    
    int step = 0;
    int check = 0;
    int* dev_check = (int *) gpu_alloc(sizeof(int));


    // Debug
    for(i=0;i<k;i++){
        for(j=0;j<dim;j++)
            printf("%lf,\t", centers[i][j]);
        printf("\n");
    }

    while (!check) {
        kmeans_on_gpu(
                    dev_points,
                    dev_centers,
                    n, k, dim,
                    dev_points_clusters,
                    dev_points_in_cluster,
                    dev_new_centers,
                    dev_check,
                    BLOCK_SIZE,
                    handle,
                    stat,
                    dev_ones,
                    dev_temp_centers);
        
        copy_from_gpu(&check, dev_check, sizeof(int));
        
        printf("Step %d Check: %d \n", step, check);
        //if (check < EPS) break;
        
        step += 1;
        //free new_centers
        // if (step == 3) break;
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
    gpu_free(dev_new_centers);
    gpu_free(dev_points);
    gpu_free(dev_points_clusters);

    stat = cublasDestroy(handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    // clear and exit
    delete_points(points);
    delete_points(centers);
    free(points_clusters);
    return 0;
}
