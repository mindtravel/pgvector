#ifndef PGVECTOR_CUDA_INTEGRATE_SCREEN_CUH
#define PGVECTOR_CUDA_INTEGRATE_SCREEN_CUH

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
 * 流水线批量查询接口（接受 device 指针）
 * 
 * 注意：所有输入数据指针必须是 device 指针，CPU-GPU 内存复制应在调用前完成
 *
 * @param d_query_batch         device 指针：query 数据，连续存储 [n_query * n_dim]
 * @param d_cluster_size         device 指针：各 cluster 中向量的数量，长度 n_total_cluster
 * @param d_cluster_vectors      device 指针：所有 cluster 向量数据，连续存储 [n_total_vectors * n_dim]
 * @param d_cluster_centers      device 指针：聚类中心数据，连续存储 [n_total_cluster * n_dim]
 * @param d_initial_indices      device 指针：初始索引数组 [n_query * n_total_cluster]，用于粗筛阶段
 *                                如果为 nullptr，则内部生成顺序索引 [0, 1, 2, ..., n_total_cluster-1]
 * @param d_topk_dist            device 指针：输出 top-k 距离矩阵 [n_query * k]
 * @param d_topk_index           device 指针：输出 top-k 索引矩阵 [n_query * k]
 * @param n_query               query 总数
 * @param n_dim                 向量维度
 * @param n_total_cluster       聚类总数
 * @param n_total_vectors       向量总数
 * @param n_probes              每个 query 精筛的 cluster 数
 * @param k                     top-k 数量
 */
void batch_search_pipeline(
    float* d_query_batch,
    int* d_cluster_size,
    float* d_cluster_vectors,
    float* d_cluster_centers,
    int* d_initial_indices,
    float* d_topk_dist,
    int* d_topk_index,
    int n_query,
    int n_dim,
    int n_total_cluster,
    int n_total_vectors,
    int n_probes,
    int k
);

// ---------------------------------------------------------
// 分离的流水线接口（用于预处理和流式计算分开调度）
// ---------------------------------------------------------

/**
 * 创建索引上下文（管理常驻显存的数据）
 * 
 * @return 返回不透明的上下文指针，用于后续所有操作
 */
extern "C" void* ivf_create_index_context();

/**
 * 销毁索引上下文
 * 
 * @param ctx_ptr 索引上下文指针
 */
extern "C" void ivf_destroy_index_context(void* ctx_ptr);

/**
 * 加载数据集到 GPU（Stage 0：初始化阶段）
 * 
 * 注意：所有输入数据指针必须是 device 指针
 * 
 * @param idx_ctx_ptr 索引上下文指针
 * @param d_cluster_size device 指针：各 cluster 中向量的数量 [n_total_clusters]
 * @param d_cluster_vectors device 指针：所有 cluster 向量数据，连续存储 [n_total_vectors * n_dim]
 * @param d_cluster_centers device 指针：聚类中心数据，连续存储 [n_total_clusters * n_dim]
 * @param n_total_clusters 聚类总数
 * @param n_total_vectors 向量总数
 * @param n_dim 向量维度
 * @return 1 如果成功，0 如果已初始化（防止重复初始化）
 */
extern "C" int ivf_load_dataset(
    void* idx_ctx_ptr,
    int* d_cluster_size,
    float* d_cluster_vectors,
    float* d_cluster_centers,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
);

/**
 * 创建查询批次上下文（管理每个Batch的临时资源和Stream）
 * 
 * 用于支持双缓冲，每个Buffer对应一个Context
 * 
 * @param max_n_query 最大查询数量
 * @param n_dim 向量维度
 * @param max_n_probes 最大probe数量
 * @param max_k 最大top-k数量
 * @param n_total_clusters 聚类总数
 * @return 返回不透明的批次上下文指针
 */
extern "C" void* ivf_create_batch_context(int max_n_query, int n_dim, int max_n_probes, int max_k, int n_total_clusters);

/**
 * 销毁查询批次上下文
 * 
 * @param ctx_ptr 批次上下文指针
 */
extern "C" void ivf_destroy_batch_context(void* ctx_ptr);

/**
 * 阶段 1: 数据预处理 (Preprocessing)
 * 
 * 职责：将Query数据上传GPU，并计算Norm。
 * 特点：利用DMA传输，不占用太多GPU Compute。
 * 
 * @param batch_ctx_ptr 批次上下文指针
 * @param query_batch_host CPU 指针：query 数据，连续存储 [n_query * n_dim]
 * @param n_query 查询数量
 */
extern "C" void ivf_pipeline_stage1_prepare(
    void* batch_ctx_ptr,
    float* query_batch_host,
    int n_query
);

/**
 * 阶段 2: 核心计算 (Compute)
 * 
 * 职责：执行粗筛和精筛。
 * 特点：计算密集。
 * 
 * @param batch_ctx_ptr 批次上下文指针
 * @param idx_ctx_ptr 索引上下文指针
 * @param n_query 查询数量
 * @param n_probes 每个query精筛的cluster数
 * @param k top-k数量
 */
extern "C" void ivf_pipeline_stage2_compute(
    void* batch_ctx_ptr,
    void* idx_ctx_ptr,
    int n_query,
    int n_probes,
    int k
);

/**
 * 获取结果 (Download)
 * 
 * 职责：将结果传回CPU
 * 
 * @param batch_ctx_ptr 批次上下文指针
 * @param topk_dist CPU 指针：输出 top-k 距离矩阵 [n_query * k]
 * @param topk_index CPU 指针：输出 top-k 索引矩阵 [n_query * k]
 * @param n_query 查询数量
 * @param k top-k数量
 */
extern "C" void ivf_pipeline_get_results(
    void* batch_ctx_ptr,
    float* topk_dist,
    int* topk_index,
    int n_query,
    int k
);

/**
 * 同步 (Wait)
 * 
 * 用于 Host 等待当前流完成
 * 
 * @param batch_ctx_ptr 批次上下文指针
 */
extern "C" void ivf_pipeline_sync_batch(void* batch_ctx_ptr);

/**
 * 流式上传接口（在 Build 过程中使用）
 */

/**
 * 初始化流式上传：分配 GPU 显存空间（不填充数据）
 */
extern void ivf_init_streaming_upload(
    void* idx_ctx_ptr,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
);

/**
 * 上传单个 Cluster 的数据 (Append Mode)
 */
extern void ivf_append_cluster_data(
    void* idx_ctx_ptr,
    int cluster_id,
    float* host_vector_data,
    int count,
    int start_offset_idx
);

/**
 * 完成流式上传：上传聚类中心，补全最后一个 Offset，计算 Norm
 */
extern void ivf_finalize_streaming_upload(
    void* idx_ctx_ptr,
    float* center_data_flat,
    int total_vectors_check
);

#endif  // PGVECTOR_CUDA_INTEGRATE_SCREEN_CUH
