#ifndef PGVECTOR_CUDA_INTEGRATE_SCREEN_CUH
#define PGVECTOR_CUDA_INTEGRATE_SCREEN_CUH
/**
 * 顶层调度入口，负责串联粗筛+精筛流水线
 */
void run_integrate_pipeline();
/**
 * 流水线批量查询接口
 *
 * @param query_batch           query 数据，指针数组，大小 [n_query]
 * @param cluster_size          各 cluster 中向量的数量，长度 n_total_cluster
 * @param cluster_data          各 cluster 在 GPU 上的起始地址数组，长度 n_total_cluster
 * @param cluster_center_data   聚类中心数据，指针数组，大小 [n_total_cluster]
 * @param topk_dist             输出：top-k 距离矩阵 [n_query * k]
 * @param topk_index            输出：top-k 索引矩阵 [n_query * k]
 * @param n_isnull              输出：每个 query 在 top-k 末尾（无有效数据）的个数
 * @param n_query               query 总数
 * @param n_dim                 向量维度
 * @param n_total_cluster       聚类总数
 * @param n_cluster_per_query   每个 query 精筛的 cluster 数（n_probes）
 * @param k                     top-k 数量
 */
void batch_search_pipeline(
    float** query_batch,
    int* cluster_size,
    float*** cluster_data,
    float** cluster_center_data,
    float** topk_dist,
    int** topk_index,
    int* n_isnull,

    int n_query,
    int n_dim,
    int n_total_cluster,
    int n_cluster_per_query,
    int k
);
#endif  // PGVECTOR_CUDA_INTEGRATE_SCREEN_CUH
