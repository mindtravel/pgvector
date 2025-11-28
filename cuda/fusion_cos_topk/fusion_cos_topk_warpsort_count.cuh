#ifndef FUSION_COS_TOPK_WARPSORT_COUNT_CUH
#define FUSION_COS_TOPK_WARPSORT_COUNT_CUH

/**
 * 紧凑数据结构：每个cluster对应4个query
 * 结构：cluster_id + query_ids[4] + probe_indices[4] = 9个int
 * 如果某个cluster的query数量超过4，会分成多个组
 * 如果少于4，用-1填充无效位置
 */
struct ClusterQueryGroup {
    int cluster_id;
    int query_ids[4];
    int probe_indices[4];
};

/**
 * 粗筛算子：计算query和cluster中心之间的余弦距离topk
 * 同时在GPU上并行统计cluster-query映射（CSR格式）
 * 并转换为紧凑格式（每个cluster对应4个query），便于传递给精筛算子
 * 
 * @param h_query_vectors query向量 [n_query][n_dim]
 * @param h_data_vectors cluster中心向量 [n_batch][n_dim]
 * @param d_data_norm cluster中心的L2范数 [n_batch]（已在GPU上）
 * @param h_topk_index 输出：topk索引 [n_query][k]
 * @param d_cluster_query_count 输出：每个cluster的query数量 [n_batch]
 * @param d_cluster_query_offset 输出：CSR格式的offset [n_batch + 1]
 * @param d_cluster_query_data 输出：CSR格式的data [total_entries]
 * @param d_cluster_query_probe_indices 输出：probe在query中的索引 [total_entries]
 * @param d_compact_groups 输出：紧凑格式的组 [n_groups]
 * @param d_n_groups 输出：实际生成的组数量 [1]
 * @param n_query query数量
 * @param n_batch cluster数量（n_total_clusters）
 * @param n_dim 向量维度
 * @param k 每个query的topk数量（n_probes）
 */
void cuda_cos_topk_warpsort_count(
    float** h_query_vectors, 
    float** h_data_vectors, 
    float* d_data_norm,
    int** h_topk_index,
    int* d_cluster_query_count,
    int* d_cluster_query_offset,
    int* d_cluster_query_data,
    int* d_cluster_query_probe_indices,
    ClusterQueryGroup* d_compact_groups,
    int* d_n_groups,
    int n_query, 
    int n_batch,
    int n_dim,
    int k
);

#endif // FUSION_COS_TOPK_WARPSORT_COUNT_CUH

