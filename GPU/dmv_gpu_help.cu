#include <stdio.h>
#include "dmv.h"

/*
 *  Utility function to get the thread ID within the
 *  global working space.
 */ 
__device__ int get_global_tid()
{
    return (gridDim.x*blockIdx.y + blockIdx.x)*blockDim.x*blockDim.y +
        blockDim.x*threadIdx.y + threadIdx.x;
}

/*
 *  Utility function to get the thread ID within the
 *  local/block working space.
 */ 
__device__ int get_local_tid()
{
    return blockDim.x*threadIdx.y + threadIdx.x;
}

/*
 *  Naive kernel
 */ 
__global__ void dmv_gpu_naive(const value_t *a, const value_t *x, value_t *y,
                              size_t n)
{
    float result=0.0;
    const int row = get_global_tid();
    int i;
    
    if(row < n)
    {
	#pragma unroll
	for(i=0; i<n; i++)
	    result += a[row*n+i]*x[i];
	y[row] = result;
    }
}

/*
 *  Coalesced memory acceses
 */
__global__ void dmv_gpu_coalesced(const value_t *a, const value_t *x,
                                  value_t *y, size_t n)
{
    float result=0.0;
    const int row = get_global_tid();
    int i;

    if(row < n)
    {
	#pragma unroll
        for(i=0; i<n; i++)
            result += a[row+i*n]*x[i];

        y[row] = result;
    }
}

/*
 *  Use of shared memory
 */
__global__ void dmv_gpu_shmem(const value_t *a, const value_t *x, value_t *y,
                              size_t n)
{
    extern __shared__ float x_sh[];
    int bl_x_ind;
    int bl_y_ind;
    int bl_l;
    float result = 0.f;
    register int i;

	bl_l = blockDim.x;
	bl_x_ind = blockIdx.x * bl_l;
	bl_y_ind = blockIdx.y * bl_l;

    x_sh[threadIdx.x] = x[bl_y_ind + threadIdx.x];

    __syncthreads();

    const int row = bl_x_ind + threadIdx.x;

    for(i=0; i<bl_l; i++) result += x_sh[i] * a[row+n*(bl_y_ind+i)];
    atomicAdd(y + row , result);
}
