/**
 * l2norm.cu
 * 求向量 l2 范数
 */
#include "l2norm.cuh"
#include <device_launch_parameters.h>
#include "pch.h"

/*
* 计算模长
* Args:
*   vectors: 原始向量组
*   vetcor_suqared_sum: 规范化后向量组的l2 norm
*   n_dim: 向量维数 
*   n_batch: 一组中向量个数 
*  
* 在向量的分量上并行计算元素的平方，再使用树形规约，最后sqrt
* 速度更快但是无法保留原始向量信息
*/
__global__ void l2_norm_kernel(float *vectors, float *vector_l2_norm, int n_batch, int n_dim) {
    extern __shared__ float shared_mem[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * n_dim + tid;

    // 计算当前向量的平方和
    float square = 0.0f;
    if (tid < n_dim) {
        square = vectors[idx] * vectors[idx];
        // printf("%d: %f %f\n", bid, vectors[idx], square);
    }

    shared_mem[tid] = square;
    __syncthreads();

    // 规约求和
    for (int s = n_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
            // int partner_idx = tid + s;
            // if (partner_idx < n_dim) { // 确保不越界
            //     shared_mem[tid] += shared_mem[partner_idx];
            // }
        }
        __syncthreads();
    }
    
    // 计算L2范数
    if (tid == 0) {
        vector_l2_norm[bid] = sqrt(shared_mem[0]);
        // printf("%d: %f\n", bid, vector_l2_norm[bid]);
    }
}
    


