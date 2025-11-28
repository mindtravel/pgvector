#ifndef FUSION_COS_TOPK_CUH
#define FUSION_COS_TOPK_CUH

#include "pch.h"

/**
 * 融合算子：纯寄存器+双调排序 实现 topk
 **/
__global__ void fusion_cos_topk_warpsort_kernel(
    float* d_query_norm, 
    float* d_data_norm, 
    float* d_inner_product, 
    int* d_index,
    int* topk_index, 
    float* topk_dist,
    int n_query, 
    int n_batch, 
    int k
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
    float** h_query_vector_group,       /*query向量*/ 
    float** h_data_vector_group,        /*data向量也就是聚类中心*/
    int** topk_index,                   /*结果：需要精筛的聚类中心的索引 [n_querys * n_probes]*/
    float** topk_cos_dist,              /*结果：需要精筛的聚类中心的距离 [n_querys * n_probes]（不一定用得上）*/
    int n_query,                        /*query数量*/
    int n_batch,                        /*data向量个数，也就是n_total_clusters*/
    int n_dim,                          /*向量维数*/
    int k                               /*查找的最近邻个数，也就是 n_probes*/
);

/**
 * 命名空间：pgvector::fusion_cos_topk_warpsort
 * 包含低级别的模板函数，用于直接操作GPU内存
 */
namespace pgvector {
namespace fusion_cos_topk_warpsort {

/**
 * 模板函数：从矩阵每一行中选取 top-k 最小或最大元素
 * 
 * 这是一个低级别的函数，直接操作GPU内存，不进行内存分配和传输
 * 
 * @param[in] d_query_norm query的L2范数 [batch_size]
 * @param[in] d_data_norm data的L2范数 [len]
 * @param[in] d_inner_product 内积矩阵 [batch_size, len]
 * @param[in] d_index 索引矩阵 [batch_size, len]
 * @param[in] batch_size 行数（批大小）
 * @param[in] len 每行的元素个数
 * @param[in] k 选取的元素个数
 * @param[out] output_vals 输出 top-k 值 [batch_size, k]
 * @param[out] output_idx 输出 top-k 对应的索引 [batch_size, k]
 * @param[in] select_min 若为 true 选取最小的 k 个，否则选取最大的 k 个
 * @param[in] stream CUDA流（可选，默认为0）
 * @return cudaError_t CUDA错误码
 */
template<typename T, typename IdxT>
cudaError_t fusion_cos_topk_warpsort(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product, const IdxT* d_index,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream = 0
);

} // namespace fusion_cos_topk_warpsort
} // namespace pgvector

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

/**
 * 流式融合余弦距离top-k计算（v2版本：优化数据上传）
 * 
 * 新设计特点：
 * 1. 只上传涉及的cluster向量（而非所有cluster）
 * 2. 使用query到cluster的映射（CSR格式）
 * 3. 预计算query和cluster向量的l2norm
 * 4. cluster向量在GPU上连续存储（只包含涉及的cluster）
 * 
 * @param d_query_group query向量 [n_query * n_dim]
 * @param d_cluster_vector 涉及的cluster向量（连续存储）[n_selected_vectors * n_dim]
 * @param d_query_cluster_offset query到cluster映射的offset [n_query+1]
 * @param d_query_cluster_data query到cluster映射的data [total_relations]
 * @param d_cluster_vector_offset 每个cluster在d_cluster_vector中的起始位置 [n_selected_clusters+1]
 * @param d_query_norm query的l2norm [n_query]
 * @param d_cluster_vector_norm cluster向量的l2norm [n_selected_vectors]
 * @param d_topk_index [out] 每个query的topk索引 [n_query * k]
 * @param d_topk_dist [out] 每个query的topk距离 [n_query * k]
 * @param n_query query数量
 * @param n_selected_clusters 涉及的cluster数量
 * @param n_selected_vectors 涉及的向量总数
 * @param n_dim 向量维度
 * @param k topk数量
 */
void cuda_cos_topk_warpsort_fine_v2(
    float* d_query_group,
    float* d_cluster_vector,
    int* d_query_index,
    
    float* d_query_norm,
    float* d_cluster_vector_norm,

    int* d_topk_index,
    float* d_topk_dist,
    
    int n_selected_clusters,
    int n_selected_vectors,
    int n_dim,
    int k
);

/**
 * 流式融合余弦距离top-k计算（v3版本：一个block处理多个query）
 * 
 * 相比v2版本的改进：
 * - 每个block处理多个query（默认8个），提高GPU利用率
 * - 减少kernel启动开销
 * - 每个warp独立处理一个query
 * 
 * @param d_query_group query向量 [n_query * n_dim]
 * @param d_cluster_vector 涉及的cluster向量（连续存储）[n_selected_vectors * n_dim]
 * @param d_query_index query索引 [n_selected_querys]
 * @param d_query_norm query的l2norm [n_query]
 * @param d_cluster_vector_norm cluster向量的l2norm [n_selected_vectors]
 * @param d_topk_index [out] 每个query的topk索引 [n_selected_querys * k]
 * @param d_topk_dist [out] 每个query的topk距离 [n_selected_querys * k]
 * @param n_selected_querys 需要处理的query数量
 * @param n_selected_vectors 涉及的向量总数
 * @param n_dim 向量维度
 * @param k topk数量
 */
void cuda_cos_topk_warpsort_fine_v3(
    float* d_query_group,
    float* d_cluster_vector,
    int* d_query_index,
    
    float* d_query_norm,
    float* d_cluster_vector_norm,

    int* d_topk_index,
    float* d_topk_dist,
    
    int n_selected_querys,
    int n_selected_vectors,
    int n_dim,
    int k
);

/**
 * 流式融合余弦距离top-k计算（v4版本：混合策略）
 * 
 * 根据数据规模动态选择最优算法：
 * - 大规模query（n_query > 100 && n_vectors > 512）：使用cublas_gemm + top-k选择
 * - 小规模query：使用v3流式实现
 */
void cuda_cos_topk_warpsort_fine_v4(
    float* d_query_group,
    float* d_cluster_vector,
    int* d_query_index,
    
    float* d_query_norm,
    float* d_cluster_vector_norm,

    int* d_topk_index,
    float* d_topk_dist,
    
    int n_selected_querys,
    int n_selected_vectors,
    int n_dim,
    int k
);

/**
 * 流式融合余弦距离top-k计算（v3_32版本：一个block处理16个query）
 * 
 * 相比v3版本的改进：
 * - 每个block处理16个query（而不是8个），进一步减少kernel启动开销
 * - 需要512个线程（16个warp，减少寄存器使用）
 */
void cuda_cos_topk_warpsort_fine_v3_32(
    float* d_query_group,
    float* d_cluster_vector,
    int* d_query_index,
    
    float* d_query_norm,
    float* d_cluster_vector_norm,

    int* d_topk_index,
    float* d_topk_dist,
    
    int n_selected_querys,
    int n_selected_vectors,
    int n_dim,
    int k
);

/**
 * 固定 probe 版本的流式融合余弦距离top-k计算（v3_fixed_probe版本）
 * 
 * 核心设计：
 * - 每个 block 处理一个 probe 的多个 query
 * - gridDim.x = n_probes（每个 block 一个 probe）
 * - gridDim.y = query batch 数量（每个 block 处理一个 probe 的一个 query batch）
 * - 利用 L2 cache：多个 query 访问相同的 probe 向量数据
 * - 输出格式：[n_query][max_probes_per_query][k]，每个probe结果写入独立位置
 * 
 * @param d_query_group query向量 [n_query * n_dim]
 * @param d_cluster_vector 所有向量数据（连续存储）[n_total_vectors * n_dim]
 * @param d_probe_vector_offset 每个probe在d_cluster_vector中的起始位置 [n_probes]
 * @param d_probe_vector_count 每个probe的向量数量 [n_probes]
 * @param d_probe_queries probe对应的query列表（CSR格式）[total_queries]
 * @param d_probe_query_offsets probe的query列表起始位置（CSR格式）[n_probes + 1]
 * @param d_probe_query_probe_indices 每个probe-query对中probe在query中的索引 [total_queries_in_probes]
 * @param d_query_norm query的l2norm [n_query]
 * @param d_cluster_vector_norm 所有向量的l2norm [n_total_vectors]
 * @param d_topk_index [out] 每个query的每个probe的topk索引 [n_query][max_probes_per_query][k]
 * @param d_topk_dist [out] 每个query的每个probe的topk距离 [n_query][max_probes_per_query][k]
 * @param n_probes probe数量
 * @param n_query query数量
 * @param max_probes_per_query 每个query的最大probe数量
 * @param n_dim 向量维度
 * @param k topk数量
 */
void cuda_cos_topk_warpsort_fine_v3_fixed_probe(
    float* d_query_group,
    float* d_cluster_vector,
    int* d_probe_vector_offset,
    int* d_probe_vector_count,
    int* d_probe_queries,
    int* d_probe_query_offsets,
    int* d_probe_query_probe_indices,
    float* d_query_norm,
    float* d_cluster_vector_norm,
    int* d_topk_index,
    float* d_topk_dist,

    float** candidate_dist,
    int** candidate_index,

    int n_query,
    int n_total_clusters,
    int n_probes,
    int n_dim,
    int k
);


#endif
