#include "kmeans_minibatch_gpu.h"


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



/* FAILED OPTIMIZATIONS

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


