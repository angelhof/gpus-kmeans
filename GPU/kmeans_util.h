#include "cublas_v2.h"

#pragma once

double** create_2D_double_array(int n, int dim);

double** init_centers_kpp(double **ps, int n, int k, int dim);

void delete_points(double** ps);

void call_create_dev_ones(double* dev_ones, int n, dim3 gpu_grid, dim3 gpu_block);

void transpose(double** src, double* dst, int n, int m);

int copy_to_gpu_constant(const double *host, size_t count);

int copy_from_gpu_constant(double *host, size_t count);

int copy_between_gpu_constant(double *host, size_t count);

void kmeans_on_gpu(
            double* dev_points,
            int n, int k, int dim,
            double* dev_points_clusters,
            double* dev_points_in_cluster,
            double* dev_new_centers,
            int* dev_check,
            int BLOCK_SIZE, 
            //CUBLAS Shit
            cublasHandle_t handle,
            cublasStatus_t stat,
            double* dev_ones,
            double* dev_temp_centers);

#ifndef MAX_CONSTANT_MEMORY
#   define MAX_CONSTANT_MEMORY 65536 / sizeof(double)
#endif
