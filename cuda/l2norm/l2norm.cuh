#ifndef L2NORM_H
#define L2NORM_H

// 全局CUDA核函数声明
__global__ void l2_norm_kernel_basic(
    float *vector_data, 
    float *vector_square_sum, 
    int n_batch, 
    int n_dim
);

__global__ void l2_norm_kernel(
    const float* __restrict__ vectors, 
    float* __restrict__ vector_l2_norm, 
    int n_batch, 
    int n_dim
);

#endif