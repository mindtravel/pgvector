/**
 * CUDA Utility Kernels Implementation
 * 
 * Common utility kernels used across multiple modules in pgvector.
 */

#include "utils.cuh"
#include <cfloat>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

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
        
        // 边界检查：确保写入位置在有效范围内
        int cluster_start = d_cluster_query_offset[cluster_id];
        int cluster_end = d_cluster_query_offset[cluster_id + 1];
        if (write_pos >= cluster_start && write_pos < cluster_end) {
            // 写入数据
            d_cluster_query_data[write_pos] = query_id;
            d_cluster_query_probe_indices[write_pos] = rank;  // rank就是probe_index_in_query
        } else {
            // 越界写入，说明有bug
            printf("[ERROR] build_cluster_query_mapping_kernel: write_pos=%d out of range [%d, %d) for cluster_id=%d\n",
                   write_pos, cluster_start, cluster_end, cluster_id);
        }
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
    int k_pos = idx % k;
    
    int candidate_pos = d_topk_index[idx];
    if (candidate_pos >= 0 && candidate_pos < max_candidates_per_query) {
        int original_idx = d_candidate_indices[query_id * max_candidates_per_query + candidate_pos];
        d_topk_index[idx] = original_idx;
    } else {
        d_topk_index[idx] = -1;
    }
}

/**
 * Host函数: 计算前缀和（用于构建CSR格式的offset数组）
 * 
 * 计算 exclusive prefix sum：offset[0] = 0, offset[i+1] = offset[i] + count[i]
 * 
 * 使用 Thrust::exclusive_scan 进行高效计算
 */
void compute_prefix_sum(
    const int* d_count,  // [n] 输入：每个元素的计数值
    int* d_offset,  // [n+1] 输出：前缀和，offset[0] = 0
    int n,  // 元素数量
    cudaStream_t stream)
{
    // 初始化 offset[0] = 0
    cudaMemsetAsync(d_offset, 0, sizeof(int), stream);
    
    // 使用 Thrust 的 exclusive_scan 计算前缀和
    // exclusive_scan: output[i] = sum(input[0..i-1])
    thrust::device_ptr<const int> count_ptr = thrust::device_pointer_cast(d_count);
    thrust::device_ptr<int> offset_ptr = thrust::device_pointer_cast(d_offset + 1);  // offset[1..n]
    
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        count_ptr,
        count_ptr + n,
        offset_ptr,
        0  // 初始值
    );
}

