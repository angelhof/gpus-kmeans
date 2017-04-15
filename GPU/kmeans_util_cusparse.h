#include "cublas_v2.h"
#include "cusparse_v2.h"

double** create_2D_double_array(int n, int dim);

double** init_centers_kpp(double **ps, int n, int k, int dim);

void delete_points(double** ps);

void call_create_dev_ones(double* dev_ones, int n, dim3 gpu_grid, dim3 gpu_block);

void transpose(double** src, double* dst, int n, int m);

void kmeans_on_gpu(
            double* dev_points,
            double* dev_centers,
            int n, int k, int dim,
            // double* dev_points_clusters,
            int* dev_points_in_cluster,
            double* dev_new_centers,
            int* dev_check,
            int block_size, 
            //CUBLAS shit
            cublasHandle_t handle,
            double* dev_ones,
            //CUSPARSE shit
            cusparseHandle_t cusparse_handle,
            // int* dev_nnzPerRow,
            double* dev_csrVal_points_clusters,
            int* dev_csrRowPtr_points_clsusters,
            int* dev_csrColInd_points_clsusters);
