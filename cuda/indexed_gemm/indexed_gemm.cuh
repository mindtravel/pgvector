#ifndef INDEXED_GEMM_CUH
#define INDEXED_GEMM_CUH

/**
 * 索引化内积计算kernel（使用连续存储）
 * 
 * 向量按聚类物理连续存储，使用连续索引访问（符合pgvector实现）。
 * 
 * 使用max_candidates_per_query作为内存布局维度（而非max_cluster_vector_count），
 * 这样可以使用实际的最大候选数，减少内存分配。
 * 使用d_num_samples来限制每个query的处理范围，避免处理无效数据。
 */
__global__ void indexed_inner_product_kernel(
    const float* __restrict__ d_query_group,
    const float* __restrict__ d_cluster_vector,
    const int* __restrict__ d_cluster_query_offset,
    const int* __restrict__ d_cluster_query_data,
    const int* __restrict__ d_cluster_vector_index,  // 每个cluster在全局向量数组中的连续起始位置
    const int* __restrict__ d_cluster_vector_num,
    float* __restrict__ d_inner_product,
    int* __restrict__ d_index,
    int* __restrict__ d_query_count,  // 每个query当前的候选数量（使用原子操作）
    const int* __restrict__ d_num_samples,  // 每个query的实际候选数量 [n_query]（用于边界检查）
    int n_query,
    int distinct_cluster_count,
    int n_dim,
    int tol_vector,
    int max_candidates_per_query  // 每个query的最大候选数（基于实际计算，用于内存布局）
);


/**
 * 流式内积计算 + top-k选择kernel（v1版本）
 * 
 * 这个kernel结合了内积计算和top-k选择，使用warp-sort queue在kernel内部流式维护topk。
 * 向量按聚类物理连续存储（符合pgvector实现）。
 * 
 * 内存优化：
 * - 直接写入最终输出 [n_query, k]，无需中间缓冲区
 * - 内存占用从 O(n_query * max_candidates) 降至 O(n_query * k)
 * 
 * @tparam Capacity warp-sort queue的容量（必须是2的幂，且 > k）
 * @tparam Ascending true表示选择最小距离（升序），false表示最大距离（降序）
 */
template<int Capacity, bool Ascending>
__global__ void indexed_inner_product_with_topk_kernel(
    const float* __restrict__ d_query_group,
    const float* __restrict__ d_cluster_vector,
    const int* __restrict__ d_cluster_query_offset,
    const int* __restrict__ d_cluster_query_data,
    const int* __restrict__ d_cluster_vector_index,
    const int* __restrict__ d_cluster_vector_num,
    const float* __restrict__ d_query_norm,
    const float* __restrict__ d_cluster_vector_norm,
    int n_query,
    int distinct_cluster_count,
    int n_dim,
    int tol_vector,
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index
);
    
#endif

