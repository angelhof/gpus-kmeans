#include "kmeans_minibatch_gpu.h"

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do
        {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val + __longlong_as_double(assumed)));
        }
    while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__global__ void fill_batch (double* points,  double* batch,  curandState* devStates, int BATCH_SIZE, int n, int dim)
{
    const unsigned int index = (gridDim.x*blockIdx.y + blockIdx.x)*blockDim.x*blockDim.y + blockDim.x*threadIdx.y + threadIdx.x;
    
    if (index < BATCH_SIZE) {
       
        int selected_index = (int) floor(curand_uniform(&devStates[index]) * (n-1));
        //Fetch Point data
        for (int i = 0, j=0; i < dim*n ; i+=n, j+=BATCH_SIZE){
            batch[index + j] = points[selected_index + i];
            // printf("Thread: %d Selected point %d Attrib: %lf Saving to Index: %d \n", 
            //     index, selected_index, points[selected_index + i], index + j);
        }
    }
};

__global__
void update_centers_minibatch_atomic(const int n, const int k, const int dim, 
                            const double* batch, 
                            double* dev_centers,
                            const int* dev_points_clusters_old, 
                            double* dev_points_in_cluster){
    int i, j, m;
    extern __shared__ double local_center_contribs[];
    const unsigned int index = (gridDim.x*blockIdx.y + blockIdx.x)*blockDim.x*blockDim.y + blockDim.x*threadIdx.y + threadIdx.x;

    //Each thread represents a batch point
    if (index < n){
        //Fetch cluster the point belongs to
        int cluster_id = dev_points_clusters_old[index];
        //Try to increase counter with atomics
        
        //atomicAdd(&dev_points_in_cluster[cluster_id], 1.0);
        int p_in_c = (int) floor(dev_points_in_cluster[cluster_id]);
        // printf("Thread: %d Points found in cluster %d : %d\n", 
        //         index, cluster_id, p_in_c);
        //calculate eta
        double eta = 1. / p_in_c;
        for (j=0, i=0; i<n*dim; j+=k, i+=n){
            double old_val = dev_centers[j + cluster_id];
            //calculate contribution
            double contrib =  eta*(batch[index + i] - old_val);
            //printf("Thread: %d cluster %d Old  %lf New %lf Eta %lf\n", 
            //    index, cluster_id, old_val, old_val + contrib, eta);
            atomicAdd(&dev_centers[j + cluster_id], contrib);
        }
    }
}



/* FAILED OPTIMIZATION KERNELS

__global__
void contribs_minibatch(const int n, const int k, const int dim, 
                            const double* batch, 
                            const int* dev_points_clusters_old, 
                            double* contributions){
    int i, j, m;
    extern __shared__ double local_center_contribs[];
    const unsigned int index = (gridDim.x*blockIdx.y + blockIdx.x)*blockDim.x*blockDim.y + blockDim.x*threadIdx.y + threadIdx.x;
    const unsigned int ltid = threadIdx.x;

    //Clear shared mem
    if (ltid < k){
        for (j=0; j<k*dim; j+=k){
            local_center_contribs[ltid + j] = 0.0;
        }
    }

    __syncthreads();
    //Each thread represents a batch point
    //At first each batch point should find its cluster and store its contribution
    //in the local_centers_contribs array
    if (index < n){
        //Fetch cluster the point belongs to
        int cluster_id = dev_points_clusters_old[index];
        
        // printf("Thread: %d Points found in cluster %d : %d\n", 
        //         index, cluster_id, p_in_c);
        //calculate eta
        for (i=0, j=0; i < n*dim; i+=n, j+=k){
            //calculate contribution
            double contrib =  batch[index + i];
            printf("Thread: %d contrib %lf\n", 
                index, contrib);
            
            atomicAdd(&local_center_contribs[j + cluster_id], contrib);
        }
    }
    __syncthreads();
    //Pass contributions to global memory
    //This will be performed from the first k threads of each block
    //This is designed for k < block_size
    
    if (ltid < k){
        int offset = k*dim*blockIdx.x;
        for (j=0; j<k*dim; j+=k){
            // printf("Block %d Thread %d Setting contribution at index %d amount %lf \n", 
            //     blockIdx.x, ltid, offset + ltid + j, local_center_contribs[ltid + j]);
            contributions[offset + ltid + j] = local_center_contribs[ltid + j];
        }
    }
}

__global__
void contribution_reduction(const int n, const int k, const int dim, 
                            double* contributions){
    int i, j, m;
    const unsigned int index = (gridDim.x*blockIdx.y + blockIdx.x)*blockDim.x*blockDim.y + blockDim.x*threadIdx.y + threadIdx.x;
    const unsigned int ltid = threadIdx.x;

    //Assuming even block and grid dims and powers of 2
    //For now it works for 1 block only
    if (index < k){
        int offset = k*dim;
        for (j=0; j<k*dim; j+=k){
            printf("Block %d Thread %d Trying to add up %d to %d: %lf + %lf \n",
                blockIdx.x, ltid,  offset + ltid + j, ltid + j, contributions[ltid + j], contributions[offset + ltid + j]);
            contributions[ltid + j] += contributions[offset + ltid + j];
        }
    }
    
}

__global__
void update_centers_minibatch(const int n, const int k, const int dim, 
                            double* dev_centers,
                            const double* dev_points_in_cluster, 
                            const double* contributions){
    int i, j, m;
    extern __shared__ double local_center_contribs[];
    const unsigned int index = (gridDim.x*blockIdx.y + blockIdx.x)*blockDim.x*blockDim.y + blockDim.x*threadIdx.y + threadIdx.x;
    const unsigned int ltid = threadIdx.x;

    ///Every thread handles one cluster
    if (ltid < k){
        int p_in_c = (int) floor(dev_points_in_cluster[ltid]);
        double eta = 1. / p_in_c;

        for (j=0; j<k*dim; j+=k){
            double contrib = contributions[ltid + j];
            //printf("Center %d Loaded contrib %lf \n", ltid,  contrib);
            
            dev_centers[ltid + j] = (1-eta) * dev_centers[ltid + j] + eta*contrib;
        }
    }
}

*/
//KMEANS ALGORITHMS

double kmeans_serial_MINIBATCH(
      double* dev_points,
      double* dev_centers,
      double *dev_new_centers,
      double *dev_points_in_cluster, 
      int n, int k, int dim,
      double* dev_points_clusters, 
      curandState* devStates, 
      cublasHandle_t handle) {

    //Create Batch
    int BATCH_SIZE = 50;
    
    //Create batch arrays
    double *batch_points, *batch_points_in_cluster, *batch_points_clusters;
    int *batch_points_clusters_old;
    batch_points = (double *) gpu_alloc(BATCH_SIZE*dim*sizeof(double));
    batch_points_clusters = (double *) gpu_alloc(BATCH_SIZE*k*sizeof(double));
    batch_points_clusters_old = (int *) gpu_alloc(BATCH_SIZE*sizeof(int));
    batch_points_in_cluster = (double *) gpu_alloc(k*sizeof(double));

    //printf("Updating Clusters in CPU \n");
    //Do the thing on the cpu
    double *cpu_centers, *cpu_new_centers, *cpu_batch_points;
    double *cpu_points,  *cpu_points_clusters, *cpu_points_in_cluster;
    int *cpu_batch_points_clusters_old;

    cpu_centers = (double*) malloc(k*dim*sizeof(double));
    cpu_new_centers = (double*) malloc(k*dim*sizeof(double));
    cpu_batch_points = (double*) malloc(BATCH_SIZE*dim*sizeof(double));
    cpu_points = (double*) malloc(n*dim*sizeof(double));
    cpu_points_in_cluster = (double*) calloc(k, sizeof(double));
    cpu_points_clusters = (double*) calloc(k*n, sizeof(double));
    cpu_batch_points_clusters_old = (int*) calloc(BATCH_SIZE, sizeof(int));

    //Copy data from GPU
    cudaMemcpy(cpu_points, dev_points, sizeof(double)*n*dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_centers, dev_centers, sizeof(double)*k*dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_new_centers, dev_centers, sizeof(double)*k*dim, cudaMemcpyDeviceToHost);
    //cudaMemcpy(cpu_batch_points, batch_points, sizeof(double)*BATCH_SIZE*dim, cudaMemcpyDeviceToHost);
    //cudaMemcpy(cpu_batch_points_clusters_old, batch_points_clusters_old, sizeof(int)*BATCH_SIZE, cudaMemcpyDeviceToHost);

    int step = 0;
    double sum = DBL_MAX;
    while (sum > EPS) {
    //while (step < 20000){
        //Init cpu_batch_points
        //Correct
        for (int index=0; index<BATCH_SIZE; index++){
            int selected_index = rand() % n;

            //Copy point
            for (int i = 0, j=0; i < dim*n ; i+=n, j+=BATCH_SIZE){
                cpu_batch_points[index + j] = cpu_points[selected_index + i];
            }
        }

        //Find Centers on CPU as well
        //Correct
        //Process
        for (int i=0; i<BATCH_SIZE; i++){
            double min = DBL_MAX;
            
            for (int j=0; j<k; j++) {
                //Calculate distance
                //lOGGING
                //printf("Distance from point %lf %lf ", cpu_batch_points[i], cpu_batch_points[i+BATCH_SIZE]);
                //printf("to center %lf %lf ", cpu_centers[j], cpu_centers[j+k]);
                double dist = squared_distance2(&cpu_batch_points[i], &cpu_centers[j], BATCH_SIZE, k, dim);
                //printf(" = %lf\n ", dist);
                if (dist < min) {
                    min=dist;
                    cpu_batch_points_clusters_old[i] = j;
                }
            }
        }

        memcpy(cpu_new_centers, cpu_centers, k*dim*sizeof(double));
        
        //Process | WORKING
        for (int i=0;i<BATCH_SIZE;i++){
            int index = cpu_batch_points_clusters_old[i];
            //printf("Batch Point %d Belongs to cluster %d with points: %d\n", i, index, cpu_points_in_cluster[index]);
            //Increment counter
            cpu_points_in_cluster[index]++;

            double eta = 1./ cpu_points_in_cluster[index];
            //printf("Eta %lf \n", eta);
            int j, m;
            //printf("Batch Point Coords: ");
            for (j=0, m=0; j<BATCH_SIZE*dim; j+=BATCH_SIZE, m+=k){
                cpu_centers[index + m] = (1.0 - eta) * cpu_centers[index + m] + eta * cpu_batch_points[i + j];
                //printf(" %lf", cpu_batch_points[i + j]);
            }
            //printf(" \n");
        }

        // //Error calculation WORKING CPU
        sum = 0.0;
        for (int i=0; i<k*dim; i++){
            double diff = cpu_centers[i] - cpu_new_centers[i];
            sum += diff*diff;
        }
        //sum = sqrt(sum);

        printf("Step %d Solution Value: %lf \n", step, sum);
        step+=1;
    }

    printf("Total Steps: %d \n", step);

    //Store Points_clusters association
    for (int i=0; i<n; i++){
            double min = DBL_MAX;
            int cluster_it_belongs;

            for (int j=0; j<k; j++) {
                //Calculate distance
                //lOGGING
                //printf("Distance from point %lf %lf ", cpu_batch_points[i], cpu_batch_points[i+BATCH_SIZE]);
                //printf("to center %lf %lf ", cpu_centers[j], cpu_centers[j+k]);
                double dist = squared_distance2(&cpu_points[i], &cpu_centers[j], n, k, dim);
                //printf(" = %lf\n ", dist);
                if (dist < min) {
                    min=dist;
                    cluster_it_belongs = j;
                }

                cpu_points_clusters[cluster_it_belongs*n + i] = 1.0;
            }
        }

    

    //Error checking
    
    // //Check for convergence with CUBLAS
    // double check = 0.0;
    // //First subtract the dev_center arrays
    // double alpha = -1.0;
    // cublasDaxpy(handle, k*dim, &alpha, dev_new_centers, 1, dev_centers, 1);
    // //Now find the norm2 of the new_centers
    // cublasDnrm2(handle, k*dim, dev_centers, 1, &check);
    
    

    //Update new centers
    // TODO: Swap pointers
    cudaMemcpy(dev_centers, cpu_new_centers, sizeof(double)*k*dim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_points_clusters, cpu_points_clusters, sizeof(double)*k*n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_new_centers, cpu_new_centers, sizeof(double)*k*dim, cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_centers, dev_new_centers, sizeof(double)*k*dim, cudaMemcpyDeviceToDevice);

    free(cpu_centers);
    free(cpu_new_centers);
    free(cpu_batch_points);
    free(cpu_points_in_cluster);
    free(cpu_points_clusters);
    free(cpu_batch_points_clusters_old);

    cudaFree(batch_points);
    cudaFree(batch_points_clusters);
    cudaFree(batch_points_in_cluster);
    cudaFree(batch_points_clusters_old);

    return 0.0;
}

double kmeans_on_gpu_MINIBATCH(
      double* dev_points,
      double* dev_centers,
      int n, int k, int dim,
      double* dev_points_clusters,
      int* dev_points_clusters_old,
      double* dev_points_in_cluster,
      double* dev_new_centers,
      int* dev_check,
      //CUBLAS Shit
      cublasHandle_t handle,
      cublasStatus_t stat,
      double* dev_ones,
      double* dev_points_help,
      double* dev_temp_centers,
      curandState* devStates, 
      //BATCH arrays
      int BATCH_SIZE, 
      double* batch_points, 
      double* batch_points_clusters, 
      int* batch_points_clusters_old)
{

    

    //Fill Batch
    dim3 gpu_grid_c((BATCH_SIZE + 32 - 1)/32, 1);
    dim3 gpu_block_c(32, 1);
    fill_batch<<<gpu_grid_c, gpu_block_c>>>(dev_points, 
                    batch_points, devStates, BATCH_SIZE, n, dim);
    //printf("BATCH Init Kernel Check: %s\n", gpu_get_last_errmsg());
    //cudaDeviceSynchronize();

    //assign Batch points to centers
    find_cluster_on_gpu3<<<gpu_grid_c, gpu_block_c, k*dim*sizeof(double)>>>(
        batch_points,
        dev_centers,
        BATCH_SIZE, k, dim,
        batch_points_clusters, 
        batch_points_clusters_old);
    //printf("find cluster for BATCH Kernel Check: %s\n", gpu_get_last_errmsg());
    //cudaDeviceSynchronize();


    //updateCenters
    //Step 1 Update Center counters
    double alpha = 1.0, beta = 1.0; //BETA 1 SHOULD MAKE THE DIFFERENCE
    cublasDgemv(handle, CUBLAS_OP_T,
                BATCH_SIZE, k,
                &alpha,
                batch_points_clusters, BATCH_SIZE,
                dev_ones, 1,
                &beta,
                dev_points_in_cluster, 1);

    //cudaMemcpy(dev_new_centers, dev_centers, k*dim*sizeof(double), cudaMemcpyDeviceToDevice);

    //Step 2 Update gradients
    update_centers_minibatch_atomic<<<gpu_grid_c, gpu_block_c, k*dim*sizeof(double)>>>(
                                    BATCH_SIZE, k, dim,
                                    batch_points,
                                    dev_centers,
                                    batch_points_clusters_old,
                                    dev_points_in_cluster);

    
    //Minibatch algorithm does not need convergence check, it works based on iterations
    //This code is just for debugging
    //Check for convergence with CUBLAS
    double check = 0.0;
    //First subtract the dev_center arrays
    // alpha = -1.0;
    // cublasDaxpy(handle, k*dim, &alpha, dev_centers, 1, dev_new_centers, 1);
    // //Now find the norm2 of the new_centers
    // cublasDnrm2(handle, k*dim, dev_new_centers, 1, &check);
    // check *= check; //Minibatch converges fast only with the squared norm :/
    
    

    //Uncomment for proper plotting
    //Store solution to new_centers
    // cudaMemcpy(dev_new_centers, dev_centers, k*dim*sizeof(double), cudaMemcpyDeviceToDevice);
    // int grid_size = (n+32-1)/32;
    // dim3 gpu_grid(grid_size, 1);
    // dim3 gpu_block(32, 1);
    // find_cluster_on_gpu3<<<gpu_grid, gpu_block, k*dim*sizeof(double)>>>(
    //     dev_points,
    //     dev_centers,
    //     n, k, dim,
    //     dev_points_clusters, 
    //     dev_points_clusters_old);


    return check;

}

