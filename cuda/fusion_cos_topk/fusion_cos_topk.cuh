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

/**
 * 适配fine_screen_top_n的融合余弦距离top-k计算
 * 
 * 这个函数专门用于精筛场景，其中query不是连续存储的，
 * 而是通过cluster_query_offset和cluster_query_data来索引。
 * 向量按聚类物理连续存储（符合pgvector实现）。
 * 
 * 使用优化的内存分配策略（参考cuVS）：
 * - max_candidates_per_query: 所有query的最大候选数（基于实际计算）
 * - d_num_samples: 每个query的实际候选数量 [n_query]
 */
void cuda_cos_topk_warpsort_fine(
    const float* d_query_group,
    const float* d_cluster_vector,
    const int* d_cluster_query_offset,
    const int* d_cluster_query_data,
    const int* d_cluster_vector_index,  // 每个cluster在全局向量数组中的连续起始位置
    const int* d_cluster_vector_num,
    const float* d_query_norm,
    const float* d_cluster_vector_norm,
    int* d_topk_index,
    float* d_topk_dist,
    int n_query,
    int distinct_cluster_count,
    int n_dim,
    int n_topn,
    int tol_vector,
    int max_candidates_per_query,  // 所有query的最大候选数（基于实际计算，而非固定上限）
    const int* d_num_samples        // 每个query的实际候选数量 [n_query]（可选，用于进一步优化）
);

/**
 * 流式融合余弦距离top-k计算（v1版本）
 * 
 * 这个函数使用流式计算方式，在kernel内部维护warp-sort queue，直接写入最终结果。
 * 内存占用从 O(n_query * max_candidates) 降至 O(n_query * k)。
 * 
 * 适用条件：
 * - k <= 256 (warp-sort容量限制)
 * - grid_dim_x == 1 (单block处理每个query的所有cluster)
 * 
 * @param d_query_group query向量组
 * @param d_cluster_vector 所有向量数据，按聚类物理连续存储
 * @param d_cluster_query_offset cluster-query倒排索引的offset数组
 * @param d_cluster_query_data cluster-query倒排索引的数据
 * @param d_cluster_vector_index 每个cluster在全局向量数组中的连续起始位置
 * @param d_cluster_vector_num 每个cluster的向量数量
 * @param d_query_norm query向量的L2范数
 * @param d_cluster_vector_norm cluster向量的L2范数
 * @param d_topk_index [out] 每个query的topk索引 [n_query][k]
 * @param d_topk_dist [out] 每个query的topk距离 [n_query][k]
 * @param n_query query总数
 * @param distinct_cluster_count 所有distinct cluster数
 * @param n_dim 向量维度
 * @param n_topn 筛选Top-N（K）个
 * @param n_total_vectors 所有向量总数
 */
void cuda_cos_topk_warpsort_fine_v1(
    const float* d_query_group,
    const float* d_cluster_vector,
    const int* d_cluster_query_offset,
    const int* d_cluster_query_data,
    const int* d_cluster_vector_index,
    const int* d_cluster_vector_num,
    const float* d_query_norm,
    const float* d_cluster_vector_norm,
    int* d_topk_index,
    float* d_topk_dist,
    int n_query,
    int distinct_cluster_count,
    int n_dim,
    int n_topn,
    int n_total_vectors
);


#endif
