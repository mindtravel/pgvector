#pragma once

// 全局CUDA核函数声明
__global__ void dotProductKernel(const float* a, const float* b, float* result, int n);
__global__ void squareKernel(const float* vec, float* result, int n);
__global__ void reduceSumKernel(const float* input, float* output, int n);
__global__ void l2_norm_kernel(float *vector_data, float *vector_square_sum, int n_batch, int n_dim);
__global__ void normalize_kernel(float *vector_data, float *vector_square_sum, int n_batch, int n_dim);
__global__ void cos_distance_kernel(
    float *query_l2_norm, 
    float *data_l2_norm, 
    float *d_cos_dist, 
    int n_query, int n_batch, int n_dim
);
