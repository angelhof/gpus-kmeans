void *gpu_alloc(size_t count);

void gpu_free(void *gpuptr);

int copy_to_gpu(const void *host, void *gpu, size_t count);

int copy_from_gpu(void *host, const void *gpu, size_t count);

const char *gpu_get_errmsg(cudaError_t err);

const char *gpu_get_last_errmsg();   
