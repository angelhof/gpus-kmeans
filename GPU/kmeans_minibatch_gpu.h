#include "cublas_v2.h"
#include "gpu_util.h"
#include <float.h>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


__global__ void fill_batch(double* points,  double* batch,
                           curandState* devStates, int BATCH_SIZE, int n, int dim);

__global__
void update_centers_minibatch_atomic(const int n, const int k, const int dim,
                                     const double* batch,
                                     double* dev_centers,
                                     const int* dev_points_clusters_old,
                                     double* dev_points_in_cluster);


/* FAILED OPTIMIZATIONS
__global__
void contribs_minibatch(const int n, const int k, const int dim,
                        const double* batch,
                        const int* dev_points_clusters_old,
                        double* contributions);

__global__
void contribution_reduction(const int n, const int k, const int dim,
                            double* contributions);

__global__
void update_centers_minibatch(const int n, const int k, const int dim,
                              double* dev_centers,
                              const double* dev_points_in_cluster,
                              const double* contributions);

*/