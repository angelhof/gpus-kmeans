#include "kmeans_sa_gpu.h"

double kmeans_on_gpu_SA(
            double* dev_points,
            double* dev_centers,
            int n, int k, int dim,
            double* dev_points_clusters,
            int* dev_points_clusters_old,
            double* dev_points_in_cluster,
            double* dev_centers_of_points,
            double* dev_new_centers,
            int* dev_check,
            dim3 gpu_grid, 
            dim3 gpu_block, 
            //CUBLAS Shit
            cublasHandle_t handle,
            cublasStatus_t stat,
            double* dev_ones,
            double* dev_points_help, 
            double* dev_temp_centers, 
            curandState* devStates, 
            double temp) {

    double alpha = 1.0, beta = 0.0;

    //Upload Temperature to constant memory
    //temp = 1.0/temp;
    //cudaMemcpyToSymbol(sa_temp, &temp, sizeof(double), 0, cudaMemcpyHostToDevice);
    
    
    //STEP 1 WITH SAKM
    /*
    SAKM_perturbation<<<gpu_grid, gpu_block, k*dim*sizeof(double)>>>(
        dev_points,
        dev_centers,
        n, k, dim,
        dev_points_clusters, 
        dev_points_clusters_old, 
        devStates);
    //printf("SA Kernel Check: %s\n", gpu_get_last_errmsg());
    cudaDeviceSynchronize();
    */

    dim3 gpu_grid_c((k + 32 - 1)/32, 1);
    dim3 gpu_block_c(32, 1);
    SAGM_perturbation<<<gpu_grid_c, gpu_block_c>>>(dev_centers, k, dim, devStates);
    //printf("SAGM Kernel Check: %s\n", gpu_get_last_errmsg());
    //cudaDeviceSynchronize();

    // assign points to clusters - step 1
    find_cluster_on_gpu<<<gpu_grid,gpu_block, k*dim*sizeof(double)>>>(
        dev_points,
        dev_centers,
        n, k, dim,
        dev_points_clusters);
    //cudaDeviceSynchronize();
    
    // update means - step 2
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                k, dim, n,
                &alpha,
                dev_points_clusters, n,
                dev_points, n,
                &beta,
                dev_new_centers, k);
    
    cublasDgemv(handle, CUBLAS_OP_T,
                n, k,
                &alpha,
                dev_points_clusters, n,
                dev_ones, 1,
                &beta,
                dev_points_in_cluster, 1);
    
    // Update centers based on counted points
    update_center_on_gpu<<<gpu_grid,gpu_block>>>(
        n, k, dim,
        dev_new_centers,
        dev_points_in_cluster);
    //cudaDeviceSynchronize();
    

    // Evaluate current solution
    double cost = evaluate_solution(dev_points, dev_new_centers, dev_points_clusters, 
                                  dev_centers_of_points, dev_points_help,
                                  n, k, dim, 
                                  gpu_grid, gpu_block, 
                                  handle, stat);
    

    //SAGM Paper notes that the cost function is the SSE (sum of squared error)
    // In order to calculate the SSE we need to 

    /*
    //Check for convergence with CUBLAS
    double check = 0.0;
    //First subtract the dev_center arrays
    alpha = -1.0;
    cudaMemcpy(dev_temp_centers, dev_centers, sizeof(double)*k*dim, cudaMemcpyDeviceToDevice);
    cublasDaxpy(handle, k*dim, &alpha, dev_new_centers, 1, dev_temp_centers, 1);
    //Now find the norm2 of the new_centers
    cublasDnrm2(handle, k*dim, dev_temp_centers, 1, &check);


    */
    return cost;
}