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
    
template<int Capacity, bool Ascending, int Dim>
__global__ void indexed_inner_product_with_topk_kernel_v2_static(
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_query_index,

    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,

    int n_selected_querys,
    int n_selected_vectors,
    int k,

    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index
);

template<int Capacity, bool Ascending>
__global__ void indexed_inner_product_with_topk_kernel_v2_generic(
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_query_index,

    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,

    int n_selected_querys,
    int n_selected_vectors,
    int n_dim,
    int k,

    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index
);

template<int Capacity, bool Ascending, int Dim, int QueriesPerBlock>
__global__ void indexed_inner_product_with_topk_kernel_v3_static(
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_query_index,

    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,

    int n_selected_querys,
    int n_selected_vectors,
    int k,

    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index
);

template<int Capacity, bool Ascending, int QueriesPerBlock>
__global__ void indexed_inner_product_with_topk_kernel_v3_generic(
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_query_index,

    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,

    int n_selected_querys,
    int n_selected_vectors,
    int n_dim,
    int k,

    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index
);

template<int Capacity, bool Ascending>
inline void launch_indexed_inner_product_with_topk_kernel_v2(
    dim3 grid,
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_query_index,
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_selected_querys,
    int n_selected_vectors,
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream = 0) {

    // 统一使用 generic 版本，支持任意维度
    indexed_inner_product_with_topk_kernel_v2_generic<Capacity, Ascending><<<grid, block, 0, stream>>>(
        d_query_group,
        d_cluster_vector,
        d_query_index,
        d_query_norm,
        d_cluster_vector_norm,
        n_selected_querys,
        n_selected_vectors,
        n_dim,
        k,
        d_topk_dist,
        d_topk_index
    );
}

/**
 * Launch函数：v3版本（一个block处理多个query）
 * 
 * @tparam Capacity warp-sort queue的容量
 * @tparam Ascending 是否升序
 * @tparam QueriesPerBlock 每个block处理的query数量（建议为8）
 * 
 * @param block 每个block的线程数（需要至少QueriesPerBlock * 32个线程）
 * @param n_dim 向量维度
 * @param n_selected_querys query总数
 * @param ... 其他参数同v2版本
 * 
 * 注意：grid配置会自动计算，grid.y = (n_selected_querys + QueriesPerBlock - 1) / QueriesPerBlock
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
inline void launch_indexed_inner_product_with_topk_kernel_v3(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_query_index,
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_selected_querys,
    int n_selected_vectors,
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream = 0) {

    /* 计算grid配置：每个block处理QueriesPerBlock个query */
    dim3 grid(1, (n_selected_querys + QueriesPerBlock - 1) / QueriesPerBlock, 1);

    // 统一使用 generic 版本，支持任意维度
    indexed_inner_product_with_topk_kernel_v3_generic<Capacity, Ascending, QueriesPerBlock><<<grid, block, 0, stream>>>(
        d_query_group,
        d_cluster_vector,
        d_query_index,
        d_query_norm,
        d_cluster_vector_norm,
        n_selected_querys,
        n_selected_vectors,
        n_dim,
        k,
        d_topk_dist,
        d_topk_index
    );
}

/**
 * Launch函数：v4版本（混合策略）
 * 
 * 根据数据规模动态选择最优算法：
 * - 大规模query（n_query > 100 && n_vectors > 512）：使用cublas_gemm
 * - 小规模query：使用v3流式实现
 * 
 * 注意：实现在indexed_gemm_v4.cu中
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_indexed_inner_product_with_topk_kernel_v4(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_query_index,
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_selected_querys,
    int n_selected_vectors,
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream = 0);

/**
 * Launch函数：v3固定probe版本（一个block处理一个probe的多个query）
 * 
 * @tparam Capacity warp-sort queue的容量
 * @tparam Ascending 是否升序
 * @tparam QueriesPerBlock 每个block处理的query数量（建议为8）
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_indexed_inner_product_with_topk_kernel_v3_fixed_probe(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_probe_queries,
    int* __restrict__ d_probe_query_offsets,
    int* __restrict__ d_probe_query_probe_indices,
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_total_clusters,  // 总的cluster数量（用于grid.x和检查probe_id）
    int n_probes,  // 每个query的probe数量（用于检查probe_index_in_query和计算输出位置）
    int max_queries_per_probe,
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream);

/**
 * Launch函数：v5 entry-based版本（按entry组织，每个entry包含一个cluster和一组query）
 * 
 * 核心设计：
 * - 每个block处理一个entry（一个cluster + 一组query，8个或4个）
 * - grid维度 = n_entry（只处理有query的cluster，避免空block）
 * - 不需要统计max_queries_per_probe，grid维度就是实际entry数量
 * 
 * @tparam Capacity warp-sort queue的容量
 * @tparam Ascending 是否升序
 * @tparam QueriesPerBlock 每个block处理的query数量（建议为8或4）
 * 
 * @param d_entry_cluster_id 每个entry对应的cluster_id [n_entry]
 * @param d_entry_query_start 每个entry的query起始位置 [n_entry]
 * @param d_entry_query_count 每个entry的query数量 [n_entry]
 * @param d_entry_queries 所有entry的query列表（扁平化）[total_queries_in_entries]
 * @param d_entry_probe_indices 每个entry-query对中probe在query中的索引 [total_queries_in_entries]
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_indexed_inner_product_with_topk_kernel_v5_entry_based(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_entry_cluster_id,  // [n_entry]
    int* __restrict__ d_entry_query_start,  // [n_entry]
    int* __restrict__ d_entry_query_count,  // [n_entry]
    int* __restrict__ d_entry_queries,  // [total_queries_in_entries]
    int* __restrict__ d_entry_probe_indices,  // [total_queries_in_entries]
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_entry,  // entry总数
    int n_probes,  // 每个query的probe数量
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream);

#endif


