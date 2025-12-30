/**
 * CUDA Utility Kernels Implementation
 * 
 * Common utility kernels used across multiple modules in pgvector.
 */

#include "utils.cuh"
#include <cfloat>

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
    int n_batch)
{
    const int query_id = blockIdx.x;
    if (query_id >= n_query) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // 每个线程处理多个索引位置（stride loop）
    for (int idx = tid; idx < n_batch; idx += block_size) {
        d_index[query_id * n_batch + idx] = idx;
    }
}

/**
 * Kernel: 生成顺序索引（用于IVF搜索的cluster索引）
 * 为每个query生成相同的顺序索引 [0, 1, 2, ..., n_total_clusters-1]
 * 
 * 线程模型：
 * - 一维 grid，每个线程处理一个索引位置
 * - total = n_query * n_total_clusters
 * - 每个query使用相同的索引序列
 */
__global__ void generate_sequential_indices_kernel(
    int* d_indices,  // [n_query * n_total_clusters] 输出：顺序索引
    int n_query,     // query数量
    int n_total_clusters  // cluster数量
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_query * n_total_clusters;
    if (idx < total) {
        int query_idx = idx / n_total_clusters;
        int cluster_idx = idx % n_total_clusters;
        d_indices[idx] = cluster_idx;  // 每个query使用相同的顺序索引
    }
}

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
    int n_total_clusters)
{
    const int query_id = blockIdx.x;
    if (query_id >= n_query) return;
    
    const int rank = threadIdx.x;
    if (rank >= k) return;
    
    int cluster_id = d_topk_index[query_id * k + rank];
    
    // 边界检查：只统计有效的cluster
    if (cluster_id >= 0 && cluster_id < n_total_clusters) {
        atomicAdd(&d_cluster_query_count[cluster_id], 1);
    }
}

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
    int n_total_clusters)
{
    const int query_id = blockIdx.x;
    if (query_id >= n_query) return;
    
    const int rank = threadIdx.x;
    if (rank >= k) return;
    
    int cluster_id = d_topk_index[query_id * k + rank];
    
    // 边界检查：只处理有效的cluster
    if (cluster_id >= 0 && cluster_id < n_total_clusters) {
        // 使用原子操作获取写入位置
        int write_pos = atomicAdd(&d_cluster_write_pos[cluster_id], 1);
        d_cluster_query_data[write_pos] = query_id;
        d_cluster_query_probe_indices[write_pos] = rank;  // rank就是probe_index_in_query
    }
}

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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    d_topk_dist_probe[idx] = FLT_MAX;
    d_topk_index_probe[idx] = -1;
}

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
) {
    int max_candidates_per_query = n_probes * k;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_query * k;
    if (idx >= total) return;
    
    int query_id = idx / k;
    
    int candidate_pos = d_topk_index[idx];
    if (candidate_pos >= 0 && candidate_pos < max_candidates_per_query) {
        int original_idx = d_candidate_indices[query_id * max_candidates_per_query + candidate_pos];
        d_topk_index[idx] = original_idx;
    } else {
        d_topk_index[idx] = -1;
    }
}

/**
 * Kernel: 填充数组为指定值
 * 
 * 线程模型：
 * - 一维 grid，每个线程处理一个元素
 * - 用于替代 thrust::fill
 */
__global__ void fill_kernel(
    float* __restrict__ d_data,
    float value,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_data[idx] = value;
}

/**
 * Kernel: 填充整数数组为指定值
 * 
 * 线程模型：
 * - 一维 grid，每个线程处理一个元素
 * - 用于替代 thrust::fill
 */
__global__ void fill_int_kernel(
    int* __restrict__ d_data,
    int value,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_data[idx] = value;
}

/**
 * Kernel: Block-level inclusive scan (Hillis-Steele算法)
 * 
 * 使用共享内存进行高效的block内前缀和计算
 * 支持多block，每个block独立计算前缀和
 */
__global__ void inclusive_scan_kernel(
    const int* __restrict__ d_input,  // [n] 输入数组
    int* __restrict__ d_output,       // [n] 输出数组
    int* __restrict__ d_block_sums,   // [n_blocks] 每个block的总和（用于后续合并）
    int n)                            // 数组长度
{
    extern __shared__ int s_data[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // 加载数据到共享内存
    if (idx < n) {
        s_data[tid] = d_input[idx];
    } else {
        s_data[tid] = 0;
    }
    __syncthreads();
    
    // Hillis-Steele inclusive scan
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid >= stride) {
            s_data[tid] += s_data[tid - stride];
        }
        __syncthreads();
    }
    
    // 保存block总和（最后一个线程）
    if (tid == blockDim.x - 1 && d_block_sums != nullptr) {
        d_block_sums[bid] = s_data[tid];
    }
    
    // 写回结果
    if (idx < n) {
        d_output[idx] = s_data[tid];
    }
}

/**
 * Kernel: 合并多block的前缀和结果
 * 
 * 将每个block的前缀和结果加上前面所有block的总和
 */
__global__ void merge_scan_blocks_kernel(
    const int* __restrict__ d_block_sums,  // [n_blocks] 每个block的总和
    int* __restrict__ d_output,            // [n] 输出数组（会被更新）
    int n,                                  // 数组长度
    int block_size)                         // 每个block的大小
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int block_id = idx / block_size;
    if (block_id > 0) {
        // 计算前面所有block的总和
        int prefix_sum = 0;
        for (int i = 0; i < block_id; i++) {
            prefix_sum += d_block_sums[i];
        }
        d_output[idx] += prefix_sum;
    }
}

/**
 * Host函数: 计算前缀和（用于构建CSR格式的offset数组）
 * 
 * 计算 inclusive prefix sum：offset[i+1] = offset[i] + count[i]
 * 然后手动设置 offset[0] = 0，得到 exclusive scan 的效果
 * 
 * 使用纯CUDA实现，不依赖Thrust
 */
void compute_prefix_sum(
    const int* d_count,  // [n] 输入
    int* d_offset,       // [n+1] 输出
    int n,               // 元素数量
    cudaStream_t stream)
{
    // 1. 手动将 offset[0] 设为 0
    cudaMemsetAsync(d_offset, 0, sizeof(int), stream);
    
    if (n <= 0) return;
    
    // 2. 计算 inclusive scan，结果写入 offset[1..n]
    const int block_size = 256;
    int n_blocks = (n + block_size - 1) / block_size;
    int shared_mem_size = block_size * sizeof(int);
    
    if (n_blocks == 1) {
        // 单block情况：直接计算，不需要block sums
        inclusive_scan_kernel<<<1, block_size, shared_mem_size, stream>>>(
            d_count,
            d_offset + 1,  // 输出到 offset[1..n]
            nullptr,       // 不需要block sums
            n
        );
    } else {
        // 多block情况：需要两阶段
        // 阶段1：每个block内部scan，并保存block总和
        int* d_block_sums = nullptr;
        cudaMalloc(&d_block_sums, n_blocks * sizeof(int));
        
        inclusive_scan_kernel<<<n_blocks, block_size, shared_mem_size, stream>>>(
            d_count,
            d_offset + 1,
            d_block_sums,
            n
        );
        
        // 阶段2：合并blocks（每个block的结果加上前面所有block的总和）
        merge_scan_blocks_kernel<<<n_blocks, block_size, 0, stream>>>(
            d_block_sums,
            d_offset + 1,
            n,
            block_size
        );
        
        cudaFree(d_block_sums);
    }
}

/**
 * Kernel: 计算每个cluster会产生多少个entry（第一阶段：计算entry数量）
 * 
 * 线程模型：
 * - gridDim.x = n_total_clusters (每个block处理一个cluster)
 * - blockDim.x = 1 (单个线程)
 * 
 * 对于每个cluster，计算其query数量，然后计算需要多少个entry（每组kQueriesPerBlock个query）
 */
__global__ void count_entries_per_cluster_kernel(
    const int* d_cluster_query_offset,  // [n_total_clusters + 1] CSR格式的offset
    int* d_entry_count_per_cluster,  // [n_total_clusters] 输出：每个cluster产生的entry数量
    int n_total_clusters,
    int kQueriesPerBlock)
{
    const int cluster_id = blockIdx.x;
    if (cluster_id >= n_total_clusters) return;
    
    int query_start = d_cluster_query_offset[cluster_id];
    int query_end = d_cluster_query_offset[cluster_id + 1];
    int n_queries = query_end - query_start;
    
    // 计算这个cluster需要多少个entry（每组kQueriesPerBlock个query）
    int n_entries = (n_queries > 0) ? (n_queries + kQueriesPerBlock - 1) / kQueriesPerBlock : 0;
    d_entry_count_per_cluster[cluster_id] = n_entries;
}

/**
 * Kernel: 构建entry数据（第二阶段：填充entry数组）
 * 
 * 线程模型：
 * - gridDim.x = n_total_clusters (每个block处理一个cluster)
 * - blockDim.x = 1 (单个线程)
 * 
 * 对于每个cluster，将其query分组为entry，并写入entry数组
 * 
 * 注意：d_entry_queries 是扁平化的数组，包含所有 entry 的 query（按 entry 顺序排列）
 * 需要计算每个 cluster 在 d_entry_queries 中的起始位置
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
    int kQueriesPerBlock)
{
    const int cluster_id = blockIdx.x;
    if (cluster_id >= n_total_clusters) return;
    
    int query_start = d_cluster_query_offset[cluster_id];
    int query_end = d_cluster_query_offset[cluster_id + 1];
    int n_queries = query_end - query_start;
    
    if (n_queries == 0) return;  // 跳过没有query的cluster
    
    int entry_start = d_entry_offset[cluster_id];
    int entry_idx = entry_start;
    
    // 计算这个cluster在entry queries数组中的起始位置
    int cluster_entry_query_start = d_entry_query_offset[cluster_id];
    int current_query_offset = cluster_entry_query_start;
    
    // 将query分组为entry（每组kQueriesPerBlock个）
    for (int batch_start = 0; batch_start < n_queries; batch_start += kQueriesPerBlock) {
        int batch_size = min(kQueriesPerBlock, n_queries - batch_start);
        
        // 写入entry元数据
        d_entry_cluster_id[entry_idx] = cluster_id;
        d_entry_query_start[entry_idx] = current_query_offset;
        d_entry_query_count[entry_idx] = batch_size;
        
        // 写入这个entry的query和probe_indices
        for (int i = 0; i < batch_size; ++i) {
            int query_idx = query_start + batch_start + i;
            d_entry_queries[current_query_offset + i] = d_cluster_query_data[query_idx];
            d_entry_probe_indices[current_query_offset + i] = d_cluster_query_probe_indices[query_idx];
        }
        
        current_query_offset += batch_size;
        entry_idx++;
    }
}
