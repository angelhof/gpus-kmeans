#include "cublas_v2.h"
#include "gpu_util.h"
#include <float.h>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#ifndef EPS
      #define EPS 1.e-6
#endif



double** create_2D_double_array(int n, int dim);

double** init_centers_kpp(double **ps, int n, int k, int dim);

void swap(double** src, double** dst);

void swap(double* src, double* dst);

void delete_points(double** ps);

void call_create_dev_ones(double* dev_ones, int n, dim3 gpu_grid, dim3 gpu_block);

void transpose(double** src, double* dst, int n, int m);

void setup_RNG_states(curandState* devStates, dim3 gpu_grid, dim3 gpu_block);

void init_point_clusters(double *dev_points, double *dev_centers,
                         int n, int k, int dim,
                         dim3 gpu_grid, dim3 gpu_block,
                         double *result_clusters, int *result_clusters_old,
                         curandState *devStates);

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
      double temp);


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
      double* dev_temp_centers);

double evaluate_solution(double* dev_points,
                         double* dev_centers,
                         double* dev_points_clusters,
                         double* dev_centers_of_points,
                         double* dev_points_help,
                         int n, int k, int dim,
                         dim3 gpu_grid, dim3 gpu_block,
                         //CUBLAS stuff
                         cublasHandle_t handle,
                         cublasStatus_t stat);