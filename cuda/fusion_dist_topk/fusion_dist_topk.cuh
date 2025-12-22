#ifndef FUSION_DIST_TOPK_CUH
#define FUSION_DIST_TOPK_CUH

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

__global__ void fusion_l2_topk_warpsort_kernel(
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
 * 命名空间：pgvector::fusion_dist_topk_warpsort
 * 包含低级别的模板函数，用于直接操作GPU内存
 */
namespace pgvector {
namespace fusion_dist_topk_warpsort {

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
cudaError_t fusion_l2_topk_warpsort(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product, const IdxT* d_index,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream = 0
);

} // namespace fusion_dist_topk_warpsort
} // namespace pgvector

/**
 * 计算共享内存大小（用于其他使用共享内存的kernel）
 **/ 
inline size_t get_shared_memory_size(int k) {
    return (k * sizeof(float) + k * sizeof(int));  /* 距离数组 + 索引数组 */
}

/**
 * Entry-based线程模型的流式融合余弦距离top-k计算
 * 
 * 核心设计：
 * - 每个 block 处理一个 entry（一个 cluster + 一组 query，8个或4个）
 * - grid维度 = n_entry（只处理有query的cluster，避免空block）
 * - 不需要统计max_queries_per_probe，grid维度就是实际entry数量
 * - 好处：不会涉及不需要的cluster，提高并行性
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
 * @param d_topk_index [out] 每个query的topk索引 [n_query][k]（最终结果，已规约）
 * @param d_topk_dist [out] 每个query的topk距离 [n_query][k]（最终结果，已规约）
 *
 * @param n_query query数量
 * @param n_total_clusters cluster数量
 * @param n_probes probe数量
 * @param n_dim 向量维度
 * @param k topk数量
 */
void cuda_cos_topk_warpsort_fine(
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