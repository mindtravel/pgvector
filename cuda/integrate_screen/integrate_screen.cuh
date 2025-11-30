#ifndef PGVECTOR_CUDA_INTEGRATE_SCREEN_CUH
#define PGVECTOR_CUDA_INTEGRATE_SCREEN_CUH
/**
 * 顶层调度入口，负责串联粗筛+精筛流水线
 */
void run_integrate_pipeline();

/**
 * 初始化常驻数据集（可在单元测试的计时之外调用）
 * 
 * 当向量数据总数小于6G时，将整个数据集传输到GPU并常驻内存
 * 如果数据已初始化且维度匹配，则直接返回，不会重复传输
 * 
 * @param cluster_size          各 cluster 中向量的数量，长度 n_total_cluster
 * @param cluster_vectors       各 cluster 的向量数据，指针数组，大小 [n_total_cluster]
 * @param cluster_center_data   聚类中心数据，指针数组，大小 [n_total_cluster]
 * @param n_total_clusters      聚类总数
 * @param n_total_vectors       向量总数
 * @param n_dim                 向量维度
 * @return true 如果数据可以常驻内存，false 如果数据太大无法常驻
 */
bool initialize_persistent_data(
    int* cluster_size,
    float*** cluster_vectors,
    float** cluster_center_data,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
);

/**
 * 清理常驻数据集
 */
void cleanup_persistent_data();

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
    int n_total_vectors,
    int n_probes,
    int k
);
#endif  // PGVECTOR_CUDA_INTEGRATE_SCREEN_CUH
