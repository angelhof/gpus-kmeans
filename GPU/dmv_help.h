#ifndef __DMV_H
#define __DMV_H

#include <stddef.h>
#include "common.h"

#if defined(__FLOAT_VALUES) || defined(__CUDACC__)
#   define MAKE_VALUE_CONSTANT(v)   v ## f
#   define VALUE_FORMAT "f"
#   define FABS    fabsf
typedef float   value_t;
#else
#   define MAKE_VALUE_CONSTANT(v)   v
#   define VALUE_FORMAT "lf"
#   define FABS    fabs
typedef double  value_t;
#endif

__BEGIN_C_DECLS

void mat_transpose(value_t **a, size_t n);
void mat_init_rand(value_t **a, size_t n, value_t max);

void vec_init(value_t *v, size_t n, value_t val);
void vec_init_rand(value_t *v, size_t n, value_t max);
int vec_equals(const value_t *v1, const value_t *v2, size_t n, value_t eps);
void vec_print(const value_t *v, size_t n);

void dmv_serial(const value_t *const *a, const value_t *x, value_t *y,
                size_t n);
void dmv_omp(const value_t *const *a, const value_t *x, value_t *y, size_t n);

__END_C_DECLS

#ifdef __CUDACC__
#   define __MAKE_KERNEL_NAME(name)   dmv_gpu ## name
#   define MAKE_KERNEL_NAME(name) __MAKE_KERNEL_NAME(name)

#   define DECLARE_GPU_KERNEL(name) \
    __global__ void MAKE_KERNEL_NAME(name)(const value_t *a,        \
                                           const value_t *x,        \
                                           value_t *y, size_t n)
#   define SHMEM_PER_BLOCK  8*1024

typedef void (*dmv_kernel_t)(const value_t *a, const value_t *x, value_t *y,
                             size_t n);

typedef struct {
    const char *name;
    dmv_kernel_t fn;
} gpu_kernel_t;

enum {
    GPU_NAIVE = 0,
    GPU_COALESCED,
    GPU_SHMEM,
    GPU_KERNEL_END
};

DECLARE_GPU_KERNEL(_naive);
DECLARE_GPU_KERNEL(_coalesced);
DECLARE_GPU_KERNEL(_shmem);

static gpu_kernel_t gpu_kernels[] = {
    {
        .name = "naive",
        .fn = MAKE_KERNEL_NAME(_naive),
    },

    {
        .name = "coalesced",
        .fn = MAKE_KERNEL_NAME(_coalesced),
    },

    {
        .name = "shmem",
        .fn = MAKE_KERNEL_NAME(_shmem),
    },
};

#endif  /* __CUDACC__ */
#endif  /* __DMV_H */
