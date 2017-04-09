#include <string.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gpu_util.h"
#include "kmeans_util_sa.h"
#include "cublas_v2.h"
#include <curand.h>
#include <curand_kernel.h>

/* gpu parameters */

//#define GRID_SIZE 16
//#define BLOCK_SIZE 256

#define DIMENSION 2

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
    double *points_clusters;

    points_clusters = (double *)calloc(n*k, sizeof(double));
    
    // GPU allocations
    double *dev_centers, *dev_points, *dev_centers_of_points;
    double *dev_points_help;
    double *dev_new_centers;
    double *dev_points_clusters;
    double *dev_points_in_cluster;
    double *dev_ones;
    //RNG CUDA States
    curandState* devStates;

    dev_centers = (double *) gpu_alloc(k*dim*sizeof(double));
    dev_points = (double *) gpu_alloc(n*dim*sizeof(double));
    dev_centers_of_points = (double *) gpu_alloc(n*dim*sizeof(double));
    dev_points_in_cluster = (double *) gpu_alloc(k*sizeof(double));
    dev_points_clusters = (double *) gpu_alloc(n*k*sizeof(double));
    dev_new_centers = (double *) gpu_alloc(k*dim*sizeof(double));
    dev_ones = (double *) gpu_alloc(n*sizeof(double));
    dev_points_help = (double *) gpu_alloc(n*sizeof(double));
    
    printf("GPU allocs done \n");
    
    call_create_dev_ones(dev_ones, n, gpu_grid, gpu_block);

    // Transpose points and centers for cublas
    // TODO: Transpose at cublas in gpu
    double * staging_points = (double*) calloc(n*dim, sizeof(double));
    double * staging_centers = (double*) calloc(k*dim, sizeof(double));
    transpose(points, staging_points, n, dim);
    transpose(centers, staging_centers, k, dim);

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

    // FIXME: For now we pass TWO matrices for centers, one normal and 
    //        one transposed. The transposed can be omitted by doing some
    //        changes in Step 1 of K-Means.
    double *dev_temp_centers;
    dev_temp_centers = (double *) gpu_alloc(k*dim*sizeof(double));

    printf("Loop Start \n");
    
    int step = 1;
    int check = 0;
    int* dev_check = (int *) gpu_alloc(sizeof(int));
    double* dev_cost = (double *) gpu_alloc(sizeof(double));

    // Debug
    for(i=0;i<k;i++){
        for(j=0;j<dim;j++)
            printf("%lf,\t", centers[i][j]);
        printf("\n");
    }
    srand(unsigned(time(NULL)));

    //SA config
    //SA starting temperature should be set so that the probablities of making moves on the very
    //first iteration should be very close to 1.
    //So this probably will need some tuning depending on the dataset
    //for the input.in start=100 and final=50, are working great
    //for dataset 100_5_2_0 from my random generator start=100000 and final=50000 work good.
    double start_temp = 80000.0;
    double temp = start_temp/log(1 + step);
    double final_temp = 50000.0;
    int eq_iterations = 2000;
    double best_cost = DBL_MAX;
    bool best_found = false;

    //SA loop
    while(temp > final_temp) {
        best_found = false;
        //printf("SA Temp: %lf \n", temp);
        //Sample solution space with SA
        for (i=0; i<eq_iterations; i++) {
            double cost = kmeans_on_gpu(
                        dev_points,
                        dev_centers,
                        n, k, dim,
                        dev_points_clusters,
                        dev_points_in_cluster,
                        dev_centers_of_points,
                        dev_new_centers,
                        dev_check,
                        dev_cost, 
                        BLOCK_SIZE,
                        handle,
                        stat,
                        dev_ones,
                        dev_points_help, 
                        dev_temp_centers, 
                        devStates, 
                        temp);

            //Check Solution and download it if its the current best
            if (best_cost - cost > EPS){
                best_cost = cost;
                best_found = true;
                printf("Found new best Solution %20.8lf at Temp %lf, instance %d \n", best_cost, temp, i);
                //Copy results from GPU 
                copy_from_gpu(staging_centers, dev_new_centers, k*dim*sizeof(double));
                copy_from_gpu(points_clusters, dev_points_clusters, n*k*sizeof(double));
                //Store results to temp_centers
                cudaMemcpy(dev_temp_centers, dev_new_centers, k*dim*sizeof(double), cudaMemcpyDeviceToDevice);
                //Try with pointer swap
                //...PENDING...
            }
        }

        //Update centers with the current best ones
        if (best_found){
            cudaMemcpy(dev_centers, dev_temp_centers, k*dim*sizeof(double), cudaMemcpyDeviceToDevice);
            //Try with pointer swap
            //...PENDING...
        }
        
        step += 1;
        //break;

        //Cooling schedule
        temp = start_temp/log(1 + step);
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
            printf("%lf ", staging_centers[j*k + i]);
            fprintf(f, "%lf ", staging_centers[j*k + i]);
        }
        printf("\n");
        fprintf(f, "\n");
    }
    fclose(f);
    
    //Store Mapping Data in case we need it
    f = fopen("point_cluster_map.out", "w");
    for (i =0;i<k;i++){
        for (j=0;j<n;j++){
            fprintf(f, "%lf ", points_clusters[i*n + j]);    
        }
        fprintf(f, "\n");
    }
    
    fclose(f);
    
    // GPU clean
    gpu_free(dev_centers);
    gpu_free(dev_new_centers);
    gpu_free(dev_points);
    gpu_free(dev_points_clusters);
    gpu_free(dev_points_in_cluster);
    gpu_free(dev_centers_of_points);

    stat = cublasDestroy(handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS destruction failed\n");
        return EXIT_FAILURE;
    }

    // clear and exit
    delete_points(points);
    delete_points(centers);
    free(points_clusters);
    return 0;
}
