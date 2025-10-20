#ifndef KERNELS_H
#define KERNELS_H

// 全局CUDA核函数声明
__global__ void l2_norm_kernel(float *vector_data, float *vector_square_sum, int n_batch, int n_dim);

__global__ void normalize_kernel(float *vector_data, float *vector_square_sum, int n_batch, int n_dim);

#endif