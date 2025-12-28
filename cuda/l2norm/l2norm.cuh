#ifndef L2NORM_H
#define L2NORM_H

#include <cuda_runtime.h>

/**
 * L2范数计算kernel版本枚举
 * 用于手动选择不同的优化策略
 */
enum L2NormVersion {
    L2NORM_AUTO = 0,           /**< 自动选择最佳版本（默认） */
    L2NORM_BASIC,              /**< 基础版本：简单的共享内存规约 */
    L2NORM_OPTIMIZED,          /**< 优化版本1：根据dim自动选择策略 */
    L2NORM_OPTIMIZED_V2,       /**< 优化版本2：简化的高效版本 */
    L2NORM_OPTIMIZED_V3        /**< 优化版本3：float4向量化加载（适用于dim是4的倍数） */
};

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

__global__ void l2_norm_kernel_optimized_v2(
    const float* __restrict__ vectors, 
    float* __restrict__ vector_l2_norm, 
    int n_batch, 
    int n_dim
);

__global__ void l2_norm_kernel_optimized_v3(
    const float* __restrict__ vectors, 
    float* __restrict__ vector_l2_norm, 
    int n_batch, 
    int n_dim
);

/**
 * 统一的L2范数计算host函数
 * 
 * 根据维度自动选择最优kernel，或使用手动指定的版本
 * 
 * @param vectors 输入向量数据（device memory）
 * @param vector_l2_squared 输出的L2范数（device memory）
 * @param n_batch 向量批次大小
 * @param n_dim 向量维度
 * @param version 算法版本（默认L2NORM_AUTO，自动选择）
 * @param stream CUDA流（可选，默认NULL）
 * 
 * 自动选择策略：
 * - n_dim <= 32: 使用warp shuffle规约（最优）
 * - 32 < n_dim <= 128: 使用warp shuffle + shared memory合并
 * - n_dim > 128: 使用优化的shared memory规约
 * - 如果dim是4的倍数且较大: 优先考虑float4向量化版本
 */
void compute_l2_norm_gpu(
    const float* vectors,
    float* vector_l2_squared,
    int n_batch,
    int n_dim,
    L2NormVersion version = L2NORM_AUTO,
    cudaStream_t stream = nullptr
);

// 全局CUDA核函数声明
__global__ void l2_squared_kernel_basic(
    float *vector_data, 
    float *vector_square_sum, 
    int n_batch, 
    int n_dim
);

__global__ void l2_squared_kernel(
    const float* __restrict__ vectors, 
    float* __restrict__ vector_l2_squared, 
    int n_batch, 
    int n_dim
);

__global__ void l2_squared_kernel_optimized_v2(
    const float* __restrict__ vectors, 
    float* __restrict__ vector_l2_squared, 
    int n_batch, 
    int n_dim
);

__global__ void l2_squared_kernel_optimized_v3(
    const float* __restrict__ vectors, 
    float* __restrict__ vector_l2_squared, 
    int n_batch, 
    int n_dim
);

/**
 * 统一的L2范数计算host函数
 * 
 * 根据维度自动选择最优kernel，或使用手动指定的版本
 * 
 * @param vectors 输入向量数据（device memory）
 * @param vector_l2_squared 输出的L2范数（device memory）
 * @param n_batch 向量批次大小
 * @param n_dim 向量维度
 * @param version 算法版本（默认L2NORM_AUTO，自动选择）
 * @param stream CUDA流（可选，默认NULL）
 * 
 * 自动选择策略：
 * - n_dim <= 32: 使用warp shuffle规约（最优）
 * - 32 < n_dim <= 128: 使用warp shuffle + shared memory合并
 * - n_dim > 128: 使用优化的shared memory规约
 * - 如果dim是4的倍数且较大: 优先考虑float4向量化版本
 */
void compute_l2_squared_gpu(
    const float* vectors,
    float* vector_l2_squared,
    int n_batch,
    int n_dim,
    L2NormVersion version = L2NORM_AUTO,
    cudaStream_t stream = nullptr
);

#endif