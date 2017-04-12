#include "kmeans_util_sa.h" 
#include <time.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

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

//Constants
__constant__ double sa_temp = 100.0;


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


#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__device__
double squared_distance_on_gpu(const double* ps, const double* center, const int n, const int k, const int dim) {
//squared_distance_on_gpu(&dev_points[i], &dev_centers[j], n, k, dim);
    double sum = 0;

    for (int i = 0, j=0; i < dim*n; i+=n,j+=k){
        double temp = center[j] - ps[i];
        sum += temp * temp;
    }

    return sum;
}

__device__
double normalized_squared_distance_on_gpu(const double* ps, const double* center, const int n, const int k, const int dim) {
    double sum = 0;
    for (int i = 0, j=0; i < dim*n; i+=n,j+=k){
        //calculate normalized distance
        double temp = center[j] - ps[i];    
        //printf("Thread %3d Mean %lf Variance %lf Temp %lf Partsum %lf \n", threadIdx.x,  mean, v1, temp * temp,  sum);
        sum += temp * temp / 2.0;
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
                         double *result_clusters) {

    double min, dist;
    int cluster_it_belongs;
    const unsigned int index = get_global_tid();
    const unsigned int thread_id = threadIdx.x;
    extern __shared__ double local_centers[];

    const unsigned int start = index;
    const unsigned int end = start + 1;

    // WARNING: Mporei na dhmiourgithei provlhma an ta threads sto block einai ligotera apo to k*dim
    if(thread_id < k*dim){
        local_centers[thread_id] = dev_centers[thread_id];
    }
    __syncthreads();

    if (index < n){
        for (int i = start; i < end; i++){
            min = DBL_MAX;
            for (int j = 0; j < k; j++){
                result_clusters[j*n + i] = 0.0;
                dist = squared_distance_on_gpu(&dev_points[i], &local_centers[j], n, k, dim);

                if (min > dist){
                    //printf("Thread: %3d Found better cluster %d \n", index, j);
                    min = dist;
                    cluster_it_belongs = j;
                }
                // cluster_it_belongs = j*(min > dist) + cluster_it_belongs*(min <= dist);
                // min = min*(min <= dist) + dist*(min > dist);
                // cluster_it_belongs = cluster_it_belongs ^ ((j ^ cluster_it_belongs) & -(min < dist));
                // min = dist ^ ((min ^ dist) & -(min < dist));
            }
            // Only 1 in the cluster it belongs and everything else 0
            // result_clusters[cluster_it_belongs*n + i] = 1.0;
            result_clusters[cluster_it_belongs*n + i] = 1.0;
            // for (int j = 0; j < k; j++){
            //     printf("result_clusters[%d][%d] = %lf --> line[%d]\n", j, i, result_clusters[j*n + i], i+2);
            // }
        }
    }
}


__global__
void find_cluster_on_gpu3(const double *dev_points, const double *dev_centers, 
                         const int n, const int k, const int dim, 
                         double *result_clusters, int *result_clusters_old) {

    double min, dist;
    int cluster_it_belongs;
    const unsigned int index = get_global_tid();
    const unsigned int thread_id = threadIdx.x;
    extern __shared__ double local_centers[];

    const unsigned int start = index;
    const unsigned int end = start + 1;

    // WARNING: Mporei na dhmiourgithei provlhma an ta threads sto block einai ligotera apo to k*dim
    if(thread_id < k*dim){
        local_centers[thread_id] = dev_centers[thread_id];
    }
    __syncthreads();

    if (index < n){
        for (int i = start; i < end; i++){
            min = DBL_MAX;
            for (int j = 0; j < k; j++){
                result_clusters[j*n + i] = 0.0;
                dist = squared_distance_on_gpu(&dev_points[i], &local_centers[j], n, k, dim);

                if (min > dist){
                    min = dist;
                    cluster_it_belongs = j;
                }
                
            }
            result_clusters[cluster_it_belongs*n + i] = 1.0;
            result_clusters_old[i] = cluster_it_belongs;
        }
    }
}

__global__
void SAGM_perturbation(double *dev_centers, 
                       const int k, const int dim, 
                       curandState *devStates) {

    const unsigned int index = get_global_tid();
    
    if (index < k) {
        //The original algorithm chooses one random cluster.
        //This is gonna be way too inefficient with a gpu so the algorithm 
        //is modified and all the centers will be perturbed simultanuously
        double delta = 0.0005;
        //Perturb Center
        //int rand_center = (int) curand_uniform(&devStates[index]) * (k-1);
        
        for (int i=0; i<k*dim; i+=k){
            double unif = delta*(2.0*curand_uniform(&devStates[index]) - 1.0)*1024.0;
            double old = dev_centers[index+i];
            dev_centers[index + i] += unif; 
            //printf("Thread: %d modified Center %d from %lf to %lf \n", index, index, old, dev_centers[index + i]);
        }
        

    }
}


__global__ void init_RNG(curandState *devStates,  unsigned long seed){
    const unsigned int index = get_global_tid();
    //Init curand for each thread
    //printf("Thread ID: %d Setting Seed %ld \n", index, seed);
    curand_init(seed, index, 0, &devStates[index]);
}

__device__ double calc_SA_prob(double de,  double dmin,  curandState* state){
    double prop = exp(-abs(de-dmin)/sa_temp);
    if (curand_uniform(state) > prop) return true;
    return prop;
}

__global__
void SAKM_perturbation(const double *dev_points, const double *dev_centers, 
                         const int n, const int k, const int dim, 
                         double *result_clusters, int *result_clusters_old, curandState *devStates) {

    double min, dist;
    int old_cluster,  new_cluster;
    const unsigned int index = get_global_tid();
    const unsigned int thread_id = threadIdx.x;
    extern __shared__ double local_centers[];

    const unsigned int start = index;
    const unsigned int end = start + 1;

    // WARNING: Mporei na dhmiourgithei provlhma an ta threads sto block einai ligotera apo to k*dim
    if(thread_id < k*dim){
        local_centers[thread_id] = dev_centers[thread_id];
    }
    __syncthreads();

    if (index < n) {
        for (int i = start; i < end; i++){
            double d_from_current = 0.0;
            //Fetch current cluster
            old_cluster = result_clusters_old[i];
            d_from_current = squared_distance_on_gpu(&dev_points[i], &local_centers[old_cluster], n, k, dim);
            
            //Find best candidate
            min = DBL_MAX;
            new_cluster = old_cluster;
            for (int j = 0; j < k; j++){
                result_clusters[j*n + i] = 0.0;
                if (j == old_cluster) continue;
                dist = squared_distance_on_gpu(&dev_points[i], &local_centers[j], n, k, dim);
                //printf("Thread: %3d Checking %d %lf vs %d %lf \n", index, old_cluster, d_from_current, j, dist);
                if (min - dist > EPS){
                    min = dist;
                    new_cluster = j;
                }
            }
            d_from_current = sqrt(d_from_current);
            min = sqrt(min);


            double prob = exp( -abs(min - d_from_current) * sa_temp );
            double unif = curand_uniform(&devStates[index]);
            
            //printf("Thread: %3d Old Center: %d %lf New Center %d %lf \n", index, old_cluster, d_from_current,  new_cluster,  min);
            //printf("Thread: %3d Temp : %lf exp(_x) %lf %lf Prob %4.3lf Unif %4.1lf Take Move:%d \n",
            //        index, 1.0/sa_temp, d_from_current, min,  prob, unif, prob > unif);
            
            if (prob > unif) {
                result_clusters[new_cluster*n + i] = 1.0;
                result_clusters_old[i] = new_cluster;
            } else {
                //Get back
                result_clusters[old_cluster*n + i] = 1.0;
                result_clusters_old[i] = old_cluster;
            }
            
             
        }
    }
}


__global__
void update_center_on_gpu(const int n, const int k, const int dim, 
                          double* dev_centers, 
                          const double* dev_points_in_cluster){
    int i, j;
    const unsigned int index = get_global_tid();

    const unsigned int start = index;
    const unsigned int end = start + 1;


    // do all numbers in k*dim threads 
    if (index < k){
        for (i = start; i < end; i++) {
            // printf("dev_points_in_cluster[%d] = %d\n", i, (int)dev_points_in_cluster[i]);
            // for (j = 0; j < dim; j++){
            //     printf("dev_centers[%d][%d] = %lf\n", i, j, dev_centers[i*dim + j]);
            // }
            if (dev_points_in_cluster[i] > 0) {
                #pragma unroll
                for (j = 0; j < dim; j++){
                    // FIXME: Two arrays here because of the transposed reslults of CUBLAS
                    //dev_temp_centers[i*dim + j] = dev_centers[j*k + i] / (int)dev_points_in_cluster[i];
                    dev_centers[j*k + i] /= dev_points_in_cluster[i];
                }
                // printf("Points in cluster: %d, %d\n", index, dev_points_in_cluster[i]);
            }
            // for (j = 0; j < dim; j++){
            //     printf("new_dev_centers[%d][%d] = %lf\n", i, j, dev_centers[i*dim + j]);
            // }
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

void swap(double* src, double* dst){
    double *temp;
    temp = src;
    src = dst;
    dst = temp;
}


__global__ void sum_distances(double* dev_points, double* dev_centers_of_points, double* dev_points_help, int n, int k, int dim){
    int index = get_global_tid();

    if(index < n){
        double dist = 0;
        for (int i = 0; i < dim*n; i+=n){
            double temp = dev_centers_of_points[index + i] - dev_points[index + i];
            dist += temp * temp;
        }

        //dist = sqrt(dist);
        dev_points_help[index] = dist;
    }
}


double evaluate_solution(double* dev_points, 
                       double* dev_centers, 
                       double* dev_points_clusters, 
                       double* dev_centers_of_points, 
                       double* dev_points_help, 
                       int n, int k, int dim, 
                       dim3 gpu_grid, dim3 gpu_block, 
                       //CUBLAS stuff
                       cublasHandle_t handle, 
                       cublasStatus_t stat){
    /*  
        The cost returned from this function is the sum of the distances
        of the points from their assigned clusters.
    */


    double cost = 0.0;
    double alpha = 1.0;
    double beta = 0.0;
    // get assigned center coords for each point
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, dim, k,
                &alpha,
                dev_points_clusters, n,
                dev_centers, k,
                &beta,
                dev_centers_of_points, n);

    alpha = -1.0;
    
    //Calculate distances and cost 
    sum_distances<<<gpu_grid, gpu_block>>>(dev_points, dev_centers_of_points, dev_points_help, n, k, dim);
    cudaDeviceSynchronize();

    //Get cost with cublas
    stat = cublasDasum(handle, n, dev_points_help, 1, &cost);

    return cost;
}

double kmeans_on_gpu(
            double* dev_points,
            double* dev_centers,
            int n, int k, int dim,
            double* dev_points_clusters,
            double* dev_points_in_cluster,
            double *dev_centers_of_points, 
            double* dev_new_centers,
            int* dev_check,
            int BLOCK_SIZE, 
            //CUBLAS Shit
            cublasHandle_t handle,
            cublasStatus_t stat,
            double* dev_ones,
            double* dev_points_help, 
            double* dev_temp_centers) {

    double alpha = 1.0, beta = 0.0;

    // Calculate grid and block sizes
    int grid_size = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 gpu_grid(grid_size, 1);
    dim3 gpu_block(BLOCK_SIZE, 1);
    
    // printf("Grid size : %dx%d\n", gpu_grid.x, gpu_grid.y);
    // printf("Block size: %dx%d\n", gpu_block.x, gpu_block.y);
    // printf("Shared memory size: %ld bytes\n", shmem_size);

    // assign points to clusters - step 1
    find_cluster_on_gpu<<<gpu_grid,gpu_block, k*dim*sizeof(double)>>>(
        dev_points,
        dev_centers,
        n, k, dim,
        dev_points_clusters);
    cudaDeviceSynchronize();
    
    // update means - step 2
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                k, dim, n,
                &alpha,
                dev_points_clusters, n,
                dev_points, n,
                &beta,
                dev_new_centers, k);
    // cudaDeviceSynchronize();

    cublasDgemv(handle, CUBLAS_OP_T,
                n, k,
                &alpha,
                dev_points_clusters, n,
                dev_ones, 1,
                &beta,
                dev_points_in_cluster, 1);
    // cudaDeviceSynchronize();

    // Update centers based on counted points
    update_center_on_gpu<<<gpu_grid,gpu_block>>>(
        n, k, dim,
        dev_new_centers,
        dev_points_in_cluster);
    cudaDeviceSynchronize();
    
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
    // cudaDeviceSynchronize();
    if (!(check > EPS)) icheck = 1;
    copy_to_gpu(&icheck, dev_check, sizeof(int));
    
    //Update new centers
    // TODO: Swap pointers
    cudaMemcpy(dev_centers, dev_new_centers, sizeof(double)*k*dim, cudaMemcpyDeviceToDevice);

    return 0.0;
}

void setup_RNG_states(curandState* devStates, dim3 gpu_grid, dim3 gpu_block){
    unsigned long seed = rand();
    //printf("Setting Seed to curandgen: %ld\n", seed);
    init_RNG<<<gpu_grid, gpu_block>>>(devStates, seed);
    //printf("SETUP RNG - CUDA Check: %s\n", gpu_get_last_errmsg());
    cudaDeviceSynchronize();    
}

void init_point_clusters(double *dev_points, double *dev_centers, 
                         int n, int k, int dim, 
                         dim3 gpu_grid, dim3 gpu_block, 
                         double *result_clusters, int *result_clusters_old, 
                         curandState *devStates){

    //Init result_cluster arrays to 0
    cudaMemset(result_clusters, 0x0, n*k*sizeof(double));
    cudaMemset(result_clusters_old, 0x0, n*sizeof(int));

    find_cluster_on_gpu3<<<gpu_grid, gpu_block, k*dim*sizeof(double)>>>(dev_points, dev_centers, 
                                                  n, k, dim, 
                                                  result_clusters, result_clusters_old);
    //printf("Init Point Clusters - CUDA Check: %s\n", gpu_get_last_errmsg());
    cudaDeviceSynchronize();
}

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
    cudaDeviceSynchronize();

    // assign points to clusters - step 1
    find_cluster_on_gpu<<<gpu_grid,gpu_block, k*dim*sizeof(double)>>>(
        dev_points,
        dev_centers,
        n, k, dim,
        dev_points_clusters);
    cudaDeviceSynchronize();
    
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
    cudaDeviceSynchronize();
    

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
