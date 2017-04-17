#include "cublas_v2.h"
#include "cusparse_v2.h"

double** create_2D_double_array(int n, int dim);

double** init_centers_kpp(double **ps, int n, int k, int dim);

void delete_points(double** ps);

void call_create_dev_ones(double* dev_ones, int n, dim3 gpu_grid, dim3 gpu_block);

void transpose(double** src, double* dst, int n, int m);

int kmeans_on_gpu(
            const double* dev_points,
            double* dev_centers,
            const int n, const int k, const int dim,
            int* dev_points_in_cluster,
            double* dev_new_centers,
            const int block_size,
            const int grid_size,
            //CUBLAS shit
            cublasHandle_t handle,
            const double* dev_ones,
            //CUSPARSE shit
            cusparseHandle_t cusparse_handle,
            double* dev_csrVal_points_clusters,
            int* dev_csrRowPtr_points_clsusters,
            int* dev_csrColInd_points_clsusters);
