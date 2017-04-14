#include "cublas_v2.h"
#include "gpu_util.h"
#include "kmeans_util_sa.h"
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

double kmeans_serial_MINIBATCH(
    double* dev_points,
    double* dev_centers,
    double *dev_new_centers,
    double *dev_points_in_cluster,
    int n, int k, int dim,
    double* dev_points_clusters,
    curandState* devStates,
    cublasHandle_t handle);

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
    int* batch_points_clusters_old);

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