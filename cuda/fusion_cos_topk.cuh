#ifndef FUSION_COS_TOPK_CUH
#define FUSION_COS_TOPK_CUH

#include "pch.h"

/**
 * 融合算子：纯寄存器+双调排序 实现 topk
 **/
__global__ void fusion_cos_topk_warpsort_kernel(
    float* d_query_norm, float* d_data_norm, float* d_inner_product, int* d_index,
    int* topk_index, float* topk_dist,
    int n_query, int n_batch, int k
);


/**
 * 融合算子：HBM显存+堆 实现 topk
 **/
__global__ void fusion_cos_topk_heap_kernel(
    float* d_query_norm, float* d_data_norm, float* d_inner_product, int* d_index,
    int* topk_index, float* topk_dist,
    int* heap_mutexes, int* heap_sizes,
    int n_query, int n_batch, int k
);

/**
 * 融合算子：共享内存+堆 实现 topk
 **/
__global__ void fusion_cos_topk_sharedmem_kernel(
    float* d_query_norm, float* d_data_norm, float* d_inner_product, int* d_index,
    int* topk_index, float* topk_dist,
    int n_query, int n_batch, int k
);

/**
 * 融合算子：纯寄存器+双调排序 实现 topk
 **/
void cuda_cos_topk_warpsort(
    float** h_query_vector_group, float** h_data_vector_group, 
    int** data_index, int** topk_index, float** topk_cos_dist,
    int n_query, int n_batch, int n_dim,
    int k /*查找的最近邻个数*/
);

/**
 * 融合算子：HBM显存+堆 实现 topk
 **/
void cuda_cos_topk_heap(
    float** h_query_vector_group, float** h_data_vector_group, 
    int** data_index, int** topk_index, float** topk_cos_dist,
    int n_query, int n_batch, int n_dim,
    int k /*查找的最近邻个数*/
);

/**
 * 融合算子：共享内存+堆 实现 topk
 **/
void cuda_cos_topk_heap_sharedmem(
    float** h_query_vector_group, float** h_data_vector_group, 
    int** data_index, int** topk_index, float** topk_cos_dist,
    int n_query, int n_batch, int n_dim,
    int k /*查找的最近邻个数*/
);

/**
 * ================= 堆实现的一些工具函数 ===================
 */

/**
 * 互斥锁加锁&解锁（内联实现，避免重复定义）
 **/
__device__ inline void mutex_lock(int* mutex) {
    while (atomicExch(mutex, 1) == 1) {
        /* 自旋等待 */ 
    }
}

__device__ inline void mutex_unlock(int* mutex) {
    atomicExch(mutex, 0);
}

/**
 * 全局内存堆操作函数
 **/
__device__ void global_heap_insert(float* heap_dist, int* heap_idx, float dist, int idx, int* heap_size, int k);
__device__ void global_heap_replace_max(float* heap_dist, int* heap_idx, float dist, int idx, int k);

/**
 * 计算共享内存大小（用于其他使用共享内存的kernel）
 **/ 
inline size_t get_shared_memory_size(int k) {
    return (k * sizeof(float) + k * sizeof(int));  /* 距离数组 + 索引数组 */
}

#endif
