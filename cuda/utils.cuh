/**
 * CUDA Utility Kernels
 * 
 * Common utility kernels used across multiple modules in pgvector.
 */

#ifndef PGVECTOR_CUDA_UTILS_CUH
#define PGVECTOR_CUDA_UTILS_CUH

/**
 * Kernel: 并行生成顺序索引
 * 为每个query生成 [0, 1, 2, ..., n_batch-1] 的索引序列
 * 
 * 线程模型：
 * - gridDim.x = n_query (每个block处理一个query)
 * - blockDim.x = min(256, n_batch) (每个block的线程数)
 * - 每个线程处理多个索引位置（如果 n_batch > blockDim.x）
 */
__global__ void generate_sequence_indices_kernel(
    int* d_index,
    int n_query,
    int n_batch
);

/**
 * Kernel: 并行统计每个cluster被多少个query使用
 * 
 * 线程模型：
 * - gridDim.x = n_query (每个block处理一个query)
 * - blockDim.x = k (每个block的线程数，对应每个query的k个probe)
 * - 每个线程处理一个query的一个probe
 * 
 * 使用原子操作来统计每个cluster的query数量
 */
__global__ void count_cluster_queries_kernel(
    const int* d_topk_index,  // [n_query * k] 粗筛结果：每个query的topk cluster索引
    int* d_cluster_query_count,  // [n_total_clusters] 输出：每个cluster的query数量
    int n_query,
    int k,
    int n_total_clusters
);

/**
 * Kernel: 构建cluster-query映射的CSR格式数据
 * 
 * 线程模型：
 * - gridDim.x = n_query (每个block处理一个query)
 * - blockDim.x = k (每个block的线程数，对应每个query的k个probe)
 * - 每个线程处理一个query的一个probe
 * 
 * 使用原子操作来确定每个cluster-query对的写入位置
 */
__global__ void build_cluster_query_mapping_kernel(
    const int* d_topk_index,  // [n_query * k] 粗筛结果
    const int* d_cluster_query_offset,  // [n_total_clusters + 1] CSR格式的offset
    int* d_cluster_query_data,  // [total_entries] CSR格式的data：query_id
    int* d_cluster_query_probe_indices,  // [total_entries] probe在query中的索引
    int* d_cluster_write_pos,  // [n_total_clusters] 每个cluster的当前写入位置（临时数组）
    int n_query,
    int k,
    int n_total_clusters
);

/**
 * Kernel: 初始化输出内存为无效值（FLT_MAX 和 -1）
 * 
 * 用于初始化 top-k 结果缓冲区，将距离初始化为 FLT_MAX，索引初始化为 -1
 * 
 * 线程模型：
 * - 一维 grid，每个线程处理一个元素
 * - total_size = n_query * n_probes * k（或类似的总元素数）
 */
__global__ void init_invalid_values_kernel(
    float* __restrict__ d_topk_dist_probe,  // [total_size] - 输出，初始化为 FLT_MAX
    int* __restrict__ d_topk_index_probe,  // [total_size] - 输出，初始化为 -1
    int total_size  // 总元素数
);

/**
 * Kernel: 映射候选索引回原始向量索引
 * 
 * 将 select_k 返回的候选数组位置索引映射回原始向量索引
 * 
 * 线程模型：
 * - 一维 grid，每个线程处理一个 top-k 结果位置
 * - total = n_query * k
 */
__global__ void map_candidate_indices_kernel(
    const int* __restrict__ d_candidate_indices,  // [n_query][n_probes * k] 原始索引数组
    int* __restrict__ d_topk_index,  // [n_query][k] - 输入是候选位置，输出是原始索引
    int n_query,
    int n_probes,
    int k
);

/**
 * Kernel: 填充数组为指定值
 * 
 * 用于替代 thrust::fill
 */
__global__ void fill_kernel(
    float* __restrict__ d_data,
    float value,
    int n);

/**
 * Kernel: 填充整数数组为指定值
 * 
 * 用于替代 thrust::fill
 */
__global__ void fill_int_kernel(
    int* __restrict__ d_data,
    int value,
    int n);

/**
 * Kernel: 前缀和计算第一阶段 - 块内前缀和和块总和
 */
__global__ void inclusive_scan_block_kernel(
    const int* __restrict__ d_input,  // [n] 输入
    int* __restrict__ d_output,  // [n] 输出：块内前缀和
    int* __restrict__ d_block_sums,  // [num_blocks] 输出：每个block的总和
    int n);

/**
 * Kernel: 前缀和计算第二阶段 - 合并块间结果
 */
__global__ void inclusive_scan_add_block_sums_kernel(
    int* __restrict__ d_output,  // [n] 输入输出：需要添加块偏移的前缀和
    const int* __restrict__ d_block_prefix_sums,  // [num_blocks] 每个block的前缀和偏移
    int n);

/**
 * Kernel: 前缀和计算（inclusive scan）- 单block版本
 * 
 * 使用并行前缀和算法，替代Thrust的inclusive_scan
 * 注意：此kernel只处理单block情况，多block请使用两阶段方法
 */
__global__ void inclusive_scan_kernel(
    const int* __restrict__ d_input,  // [n] 输入
    int* __restrict__ d_output,  // [n] 输出
    int n);

/**
 * Host函数: 计算前缀和（用于构建CSR格式的offset数组）
 * 
 * 计算 exclusive prefix sum：offset[0] = 0, offset[i+1] = offset[i] + count[i]
 * 
 * 使用自定义的inclusive_scan kernel替代Thrust
 * 
 * @param d_count 输入：每个元素的计数值 [n]（GPU内存）
 * @param d_offset 输出：前缀和结果 [n+1]（GPU内存），offset[0] = 0
 * @param n 元素数量
 * @param stream CUDA流（可选，默认为0）
 */
void compute_prefix_sum(
    const int* d_count,  // [n] 输入：每个元素的计数值
    int* d_offset,  // [n+1] 输出：前缀和，offset[0] = 0
    int n,  // 元素数量
    cudaStream_t stream = 0
);

/**
 * Kernel: 计算每个cluster会产生多少个entry（第一阶段：计算entry数量）
 * 
 * 线程模型：
 * - gridDim.x = n_total_clusters (每个block处理一个cluster)
 * - blockDim.x = 1 (单个线程)
 */
__global__ void count_entries_per_cluster_kernel(
    const int* d_cluster_query_offset,  // [n_total_clusters + 1] CSR格式的offset
    int* d_entry_count_per_cluster,  // [n_total_clusters] 输出：每个cluster产生的entry数量
    int n_total_clusters,
    int kQueriesPerBlock);

/**
 * Kernel: 构建entry数据（第二阶段：填充entry数组）
 * 
 * 线程模型：
 * - gridDim.x = n_total_clusters (每个block处理一个cluster)
 * - blockDim.x = 1 (单个线程)
 */
__global__ void build_entry_data_kernel(
    const int* d_cluster_query_offset,  // [n_total_clusters + 1] CSR格式的offset
    const int* d_cluster_query_data,  // [total_entries] CSR格式的data：query_id
    const int* d_cluster_query_probe_indices,  // [total_entries] probe在query中的索引
    const int* d_entry_offset,  // [n_total_clusters + 1] entry的offset（CSR格式）
    const int* d_entry_query_offset,  // [n_total_clusters + 1] 每个cluster在d_entry_queries中的起始位置
    int* d_entry_cluster_id,  // [n_entry] 输出：每个entry对应的cluster_id
    int* d_entry_query_start,  // [n_entry] 输出：每个entry的query起始位置（在d_entry_queries中的偏移）
    int* d_entry_query_count,  // [n_entry] 输出：每个entry的query数量
    int* d_entry_queries,  // [total_queries_in_entries] 输出：所有entry的query列表
    int* d_entry_probe_indices,  // [total_queries_in_entries] 输出：所有entry的probe_indices
    int n_total_clusters,
    int kQueriesPerBlock);

#endif // PGVECTOR_CUDA_UTILS_CUH
