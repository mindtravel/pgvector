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
 * Launch函数：entry-based版本（一个block处理一个entry：一个cluster + 一组query）
 * 
 * @tparam Capacity warp-sort queue的容量
 * @tparam Ascending 是否升序
 * @tparam QueriesPerBlock 每个entry包含的query数量（建议为8）
 * 
 * Entry-based线程模型：
 * - grid维度 = n_entry（只处理有query的cluster）
 * - 每个block处理一个entry（一个cluster + 一组query）
 * - 避免为没有query的cluster分配空block，提高并行性
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_indexed_inner_product_with_cos_topk_kernel(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_entry_cluster_id,  // [n_entry] 每个entry对应的cluster_id
    int* __restrict__ d_entry_query_start,  // [n_entry] 每个entry的query起始位置
    int* __restrict__ d_entry_query_count,  // [n_entry] 每个entry的query数量
    int* __restrict__ d_entry_queries,  // [total_queries_in_entries] 所有entry的query列表
    int* __restrict__ d_entry_probe_indices,  // [total_queries_in_entries] 所有entry的probe_indices
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_entry,  // entry总数
    int n_probes,  // 每个query的probe数量
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream);

template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_indexed_inner_product_with_l2_topk_kernel(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_entry_cluster_id,  // [n_entry] 每个entry对应的cluster_id
    int* __restrict__ d_entry_query_start,  // [n_entry] 每个entry的query起始位置
    int* __restrict__ d_entry_query_count,  // [n_entry] 每个entry的query数量
    int* __restrict__ d_entry_queries,  // [total_queries_in_entries] 所有entry的query列表
    int* __restrict__ d_entry_probe_indices,  // [total_queries_in_entries] 所有entry的probe_indices
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_entry,  // entry总数
    int n_probes,  // 每个query的probe数量
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream);
#endif


