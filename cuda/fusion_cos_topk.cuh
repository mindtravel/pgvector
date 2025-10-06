#ifndef FUSION_COS_TOPK_CUH
#define FUSION_COS_TOPK_CUH

#include "pch.h"

/**
 * 融合算子：内积->余弦距离->topk
 **/
__global__ void fusion_cos_topk_kernel(
    float* d_query_norm, float* d_data_norm, float* d_inner_product, int* d_index,
    int* topk_index, float* topk_dist,
    int n_query, int n_batch, int k
);

/**
 * 互斥锁加锁&解锁
 **/
__device__ void mutex_lock(int* mutex);
__device__ void mutex_unlock(int* mutex);

/**
 * 共享内存堆操作函数
 **/
__device__ void shared_heap_insert(float* heap_dist, int* heap_idx, float dist, int idx, int* heap_size, int k);
__device__ void shared_heap_replace_max(float* heap_dist, int* heap_idx, float dist, int idx, int k);

/**
 * 计算共享内存大小
 **/ 
inline size_t get_shared_memory_size(int k) {
    return (k * sizeof(float) + k * sizeof(int));  /* 距离数组 + 索引数组 */
}

#endif
