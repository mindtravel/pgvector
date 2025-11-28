/**
 * Warp-Sort Top-K Implementation for pgvector
 * 
 * Based on RAFT (RAPIDS AI) warp-sort implementation:
 * - raft/cpp/include/raft/matrix/detail/select_warpsort.cuh
 * - raft/cpp/include/raft/util/bitonic_sort.cuh
 * 
 * This implementation provides GPU-accelerated top-k selection using
 * warp-level primitives and bitonic sorting networks.
 * 
 * Key features:
 * - Support for k up to 256 (kMaxCapacity)
 * - Warp-level parallelism using shuffle operations
 * - Register-based storage for minimal memory overhead
 * - Bitonic merge network for efficient sorting
 * 
 * Copyright (c) 2024, pgvector
 * Adapted from RAFT (Apache 2.0 License)
 */

#include <limits>
#include <type_traits>
#include <math_constants.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>

#include "fusion_cos_topk.cuh"
#include "../unit_tests/common/test_utils.cuh"
#include "pch.h"
#include "warpsortfilter/warpsort_utils.cuh"
#include "warpsortfilter/warpsort.cuh"

#define ENABLE_CUDA_TIMING 0

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
        
        // 写入数据
        d_cluster_query_data[write_pos] = query_id;
        d_cluster_query_probe_indices[write_pos] = rank;  // rank就是probe_index_in_query
    }
}

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
 * Kernel: 将CSR格式的cluster-query映射转换为紧凑格式（每个cluster对应4个query）
 * 
 * 线程模型：
 * - gridDim.x = n_total_clusters (每个block处理一个cluster)
 * - blockDim.x = 1 (每个block一个线程)
 * - 每个线程处理一个cluster，将其query分组（每组4个）
 */
__global__ void convert_to_compact_groups_kernel(
    const int* d_cluster_query_offset,  // [n_total_clusters + 1] CSR格式的offset
    const int* d_cluster_query_data,  // [total_entries] CSR格式的data：query_id
    const int* d_cluster_query_probe_indices,  // [total_entries] probe在query中的索引
    ClusterQueryGroup* d_compact_groups,  // [n_groups] 输出：紧凑格式的组
    int* d_group_count,  // [1] 输出：实际生成的组数量
    int n_total_clusters)
{
    const int cluster_id = blockIdx.x;
    if (cluster_id >= n_total_clusters) return;
    
    int start = d_cluster_query_offset[cluster_id];
    int end = d_cluster_query_offset[cluster_id + 1];
    int n_queries = end - start;
    
    if (n_queries == 0) return;  // 跳过没有query的cluster
    
    // 计算这个cluster需要多少个组（每组4个query）
    int n_groups = (n_queries + 3) / 4;  // 向上取整
    
    // 为每个组填充数据
    for (int g = 0; g < n_groups; g++) {
        int group_start = start + g * 4;
        int group_end = min(group_start + 4, end);
        int group_size = group_end - group_start;
        
        // 使用原子操作获取组写入位置
        int group_idx = atomicAdd(d_group_count, 1);
        
        // 填充组数据
        d_compact_groups[group_idx].cluster_id = cluster_id;
        
        // 填充query_ids和probe_indices
        for (int i = 0; i < 4; i++) {
            if (i < group_size) {
                int pos = group_start + i;
                d_compact_groups[group_idx].query_ids[i] = d_cluster_query_data[pos];
                d_compact_groups[group_idx].probe_indices[i] = d_cluster_query_probe_indices[pos];
            } else {
                // 用-1填充无效位置
                d_compact_groups[group_idx].query_ids[i] = -1;
                d_compact_groups[group_idx].probe_indices[i] = -1;
            }
        }
    }
}

namespace pgvector {
namespace fusion_cos_topk_warpsort {

using namespace warpsort_utils;
using namespace warpsort;

// ============================================================================
// Public API: Top-K Selection Kernel
// ============================================================================

/**
 * 从矩阵每一行中选取 top-k 最小或最大元素。
 * 
 * 每个 CUDA block 处理一行。每个 block 内的 warp 独立地使用 WarpSortFiltered 算法完成排序筛选。
 * 
 * @param[in] d_query_norm        输入的query L2范数内积矩阵，形状为 [n_query]
 * @param[in] d_data_norm        输入的data L2范数内积矩阵，形状为 [n_batch]
 * @param[in] d_inner_product        输入的内积矩阵，形状为 [n_query, n_batch]
 * @param[in] d_index        输入的索引，形状为 [n_query, n_batch]
 * @param[in] batch_size   行数（批大小）
 * @param[in] len          每行的元素个数
 * @param[in] k            选取的元素个数
 * @param[out] output_vals 输出 top-k 值，形状为 [n_query, k]
 * @param[out] output_idx  输出 top-k 对应的索引，形状为 [n_query, k]
 * @param[in] select_min   若为 true 选取最小的 k 个，否则选取最大的 k 个
 */
template<int Capacity, bool Ascending, typename T, typename IdxT>
__global__ void fusion_cos_topk_warpsort_kernel(
    const T* __restrict__ d_query_norm,
    const T* __restrict__ d_data_norm,
    const T* __restrict__ d_inner_product,
    const IdxT* __restrict__ d_index,
    int batch_size,
    int len,
    int k,
    T* __restrict__ output_vals,
    IdxT* __restrict__ output_idx)
{
    const int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = laneId();
    const int n_warps = blockDim.x / kWarpSize; /* 一共参与WarpSort的数量（目前都是1） */
    
    /* 在一个warp中，建立一个长度为k的队列 */
    WarpSortFiltered<Capacity, Ascending, T, IdxT> queue(k); 
    
    float query_norm = d_query_norm[row];
    
    /* 获取dummy值，用于无效元素（确保所有线程都参与同步）*/
    /* 使用 WarpSort 基类的静态方法获取正确的dummy值 */
    using BaseWarpSort = WarpSort<Capacity, Ascending, T, IdxT>;
    const T dummy_val = BaseWarpSort::kDummy();

    /* 按照 laneId 访问数据 */
    const T* row_inner_product = d_inner_product + row * len;
    const IdxT* row_index = d_index + row * len;
    
    /* 
     * 关键修复：使用固定次数的循环，确保所有线程执行相同次数的迭代
     * 这对于 WarpSortFiltered 的 any() 和 __any_sync() 同步是必需的
     * 
     * 问题：原来的条件循环 `for (int i = ...; i < len; i += ...)` 导致不同线程
     * 执行不同次数的迭代，使得 queue.add() 的调用不同步，导致 __any_sync() 死锁
     * 
     * 解决方案：计算最大迭代次数，所有线程执行相同次数的循环，
     * 每个迭代都同步调用 queue.add()（无论是否有有效数据）
     */
    
    /* 确保所有线程都执行到这里 */
    __syncwarp();
    
    /* 计算最大迭代次数：ceil(len / (n_warps * kWarpSize)) */
    int max_iter = (len + n_warps * kWarpSize - 1) / (n_warps * kWarpSize);
    
    for (int iter = 0; iter < max_iter; iter++) {
        /* 在每个迭代开始时同步 */
        __syncwarp();
        
        /* 计算当前迭代处理的索引 */
        int i = warp_id * kWarpSize + lane + iter * n_warps * kWarpSize;
        bool has_data = (i < len);
        
        if (has_data) {
            float data_norm = d_data_norm[i];
            float inner_product = row_inner_product[i];
            IdxT index = row_index[i];
            
            /* 边界检查：避免无效数据 */
            /* 对于索引类型，如果为负数（有符号）或特殊标记值，视为无效 */
            bool is_valid_index = true;
            if constexpr (std::is_signed_v<IdxT>) {
                is_valid_index = (index >= 0);
            }
            
            if (data_norm >= 1e-6f && is_valid_index) {
                float cos_similarity = inner_product / (query_norm * data_norm);
                float cos_distance = 1.0f - cos_similarity;
                queue.add(cos_distance, index);
            } else {
                /* 无效数据，添加dummy值 */
                queue.add(dummy_val, IdxT{});
            }
        } else {
            /* 超出范围，添加dummy值以确保所有线程同步 */
            queue.add(dummy_val, IdxT{});
        }
    }
    
    /* 确保所有线程都完成了循环 */
    __syncwarp();
    
    /* 把 buffer 中剩余数合并到 queue 中 */
    queue.done();
    
    /* 将 queue 中的数存储到显存中（所有线程都要调用）*/
    if (warp_id == 0) {
        T* row_out_val = output_vals + row * k;
        IdxT* row_out_idx = output_idx + row * k;
        queue.store(row_out_val, row_out_idx);
    }
}

/**
 * Host function to launch top-k selection.
 * Automatically chooses appropriate capacity based on k.
 */
template<typename T, typename IdxT>
cudaError_t fusion_cos_topk_warpsort(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product, const IdxT* d_index,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream = 0
)
{
    if (k > kMaxCapacity) {
        return cudaErrorInvalidValue;
    }
    
    /* 
     * 选择合适的 Capacity
     * 
     * WarpSortFiltered 需要 buffer 空间，因此：
     * - Capacity 必须 > k（不能等于 k）
     * - 最小使用 64（确保 kMaxArrLen >= 2）
     * - 选择最小的满足 Capacity > k 的 2 的幂
     */
    int capacity = 32;  /* 最小使用 32 */
    while (capacity < k) capacity <<= 1;  /* 注意：必须 > k，不能等于 */
    
    dim3 block(32);  /* 使用32线程（单个warp）*/
    dim3 grid(batch_size);
    
    /* 模板的非类型参数必须是常量，所以只能用这一系列分支来使用不同尺寸的函数 */
    if (select_min) {
        if (capacity <= 32) {
            fusion_cos_topk_warpsort_kernel<64, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        } else if (capacity <= 64) {
            fusion_cos_topk_warpsort_kernel<128, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        } else {
            fusion_cos_topk_warpsort_kernel<256, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        }
    } else {
        if (capacity <= 32) {
            fusion_cos_topk_warpsort_kernel<64, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index,
                batch_size, len, k, 
                output_vals, output_idx
            );
        } else if (capacity <= 64) {
            fusion_cos_topk_warpsort_kernel<128, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        } else {
            fusion_cos_topk_warpsort_kernel<256, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        }
    }
    
    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t fusion_cos_topk_warpsort<float, int>(
    const float*, const float*, const float*, const int*, int, int, int, float*, int*, bool, cudaStream_t);

template cudaError_t fusion_cos_topk_warpsort<float, uint32_t>(
    const float*, const float*, const float*, const uint32_t*, int, int, int, float*, uint32_t*, bool, cudaStream_t);

} // namespace warpsort
} // namespace pgvector


void cuda_cos_topk_warpsort_count(
    float** h_query_vectors, 
    float** h_data_vectors, 

    float* d_data_norm,

    int** h_topk_index,  // 输出：topk索引 [n_query][k]
    int* d_cluster_query_count,  // 输出：每个cluster的query数量 [n_batch]
    int* d_cluster_query_offset,  // 输出：CSR格式的offset [n_batch + 1]
    int* d_cluster_query_data,  // 输出：CSR格式的data [total_entries]
    int* d_cluster_query_probe_indices,  // 输出：probe在query中的索引 [total_entries]
    
    // 紧凑格式输出（每个cluster对应4个query）
    ClusterQueryGroup* d_compact_groups,  // 输出：紧凑格式的组 [n_groups]
    int* d_n_groups,  // 输出：实际生成的组数量 [1]
    
    int n_query, 
    int n_batch,  // n_total_clusters
    int n_dim,
    int k  // n_probes
){ 
    /**
    * 对一个batch的查询向量，找出余弦距离最近的topk，返回索引矩阵
    * 同时在GPU上并行统计cluster-query映射（CSR格式）
    * 并转换为紧凑格式（每个cluster对应4个query），便于传递给精筛算子
    * 注意：不返回dist，只作为局部变量
    **/

    float alpha = 1.0f; 
    float beta = 0.0f;

    const int NUM_STREAMS = 0; // cuda流的数量
    bool query_copied = false; // query常驻显存

    dim3 queryDim(n_query);
    dim3 dataDim(n_batch);
    dim3 vectorDim(n_dim);

    cudaStream_t streams[NUM_STREAMS];
    
    size_t size_query = n_query * n_dim * sizeof(float);
    size_t size_data = n_batch * n_dim * sizeof(float);
    size_t size_dist = n_query * n_batch * sizeof(float);
    size_t size_index = n_query * n_batch * sizeof(int);
    size_t size_topk_dist = n_query * k * sizeof(float);  // 局部变量，不返回
    size_t size_topk_idx = n_query * k * sizeof(int);

    // cuBLAS句柄
    cublasHandle_t handle;
    // cublasSetStream(handle, streams[0]); 

    // 分配设备内存
    float *d_query_vectors, *d_data_vectors, *d_inner_product, *d_topk_cos_dist,  // dist作为局部变量
        *d_query_norm;
    int *d_index, *d_topk_index;
    int *d_cluster_write_pos;  // 临时数组：每个cluster的写入位置
    {
        CUDATimer timer_manage("GPU Memory Allocation", ENABLE_CUDA_TIMING);

        cudaMalloc(&d_query_vectors, size_query);
        cudaMalloc(&d_data_vectors, size_data);
        cudaMalloc(&d_inner_product, size_dist);/*存储各个query需要查找的data向量的距离*/
        cudaMalloc(&d_index, size_index);/*存储各个query需要查找的data向量的索引*/
        cudaMalloc(&d_topk_cos_dist, size_topk_dist);/*存储topk距离（局部变量，不返回）*/
        cudaMalloc(&d_topk_index, size_topk_idx);/*存储topk索引*/

        cudaMalloc(&d_query_norm, n_query * sizeof(float)); /*存储query的l2 Norm*/
        
        // 分配临时数组用于构建cluster-query映射
        cudaMalloc(&d_cluster_write_pos, n_batch * sizeof(int));  // 每个cluster的写入位置
        
        // 初始化组数量计数器
        cudaMemset(d_n_groups, 0, sizeof(int));

        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        }

        cublasCreate(&handle);
    }

    // 复制数据到设备
    {
        // COUT_ENDL("begin data transfer");

        CUDATimer timer_trans1("H2D Data Transfer", ENABLE_CUDA_TIMING);
        // 复制查询向量，然后常驻
        if(query_copied == false){
            cudaMemcpy2D(
                d_query_vectors,
                n_dim * sizeof(float),
                h_query_vectors[0],
                n_dim * sizeof(float),
                n_dim * sizeof(float),
                n_query,
                cudaMemcpyHostToDevice
            );            
            query_copied = true;
        }

        /* 复制data向量 */
        cudaMemcpy2D(
            d_data_vectors,
            n_dim * sizeof(float),
            h_data_vectors[0],
            n_dim * sizeof(float),
            n_dim * sizeof(float),
            n_batch,
            cudaMemcpyHostToDevice
        );
        // cudaMemcpy(d_data_vectors, h_data_vectors, size_data, cudaMemcpyHostToDevice);        

        /* 使用 CUDA kernel 并行生成顺序索引 [0, 1, 2, ..., n_batch-1] */
        // 线程模型：每个block处理一个query，每个block使用256个线程（或更少）
        const int threads_per_block = 256;
        dim3 block_dim((n_batch < threads_per_block) ? n_batch : threads_per_block);
        dim3 grid_dim(n_query);
        
        generate_sequence_indices_kernel<<<grid_dim, block_dim>>>(
            d_index, n_query, n_batch);
        // CHECK_CUDA_ERRORS;

        /* 初始化距离数组（为一个小于-1的负数） */
        thrust::fill(
            thrust::device_pointer_cast(d_topk_cos_dist),/*使用pointer_cast不用创建临时对象*/
            thrust::device_pointer_cast(d_topk_cos_dist) + (n_query * k),  /* 使用元素数量而非字节数 */
            FLT_MAX
        );
    }

    /* 核函数执行 */
    {
        // COUT_ENDL("begin_kernel");

        CUDATimer timer_compute("Kernel Execution: l2 Norm + matrix multiply", ENABLE_CUDA_TIMING);

        l2_norm_kernel<<<queryDim, vectorDim, n_dim * sizeof(float)>>>(
            d_query_vectors, d_query_norm, 
            n_query, n_dim
        );

        // COUT_ENDL("finish l2 norm");

        // table_cuda_2D("topk cos distance", d_topk_cos_dist, n_query, k);
       
        // table_cuda_1D("query_norm", d_query_norm, n_query);
        // table_cuda_1D("data_norm", d_data_norm, n_batch);
        // table_cuda_2D("data vectors", d_data_vectors, n_batch, n_dim);

        /**
        * 使用cuBLAS进行矩阵乘法
        * cuBLAS默认使用列主序，leading dimension是行数
        * */ 
       cublasSgemm(handle, 
            CUBLAS_OP_T, CUBLAS_OP_N, 
            n_batch, n_query, n_dim,                   
            &alpha, 
            d_data_vectors, n_dim,            
            d_query_vectors, n_dim,               
            &beta, 
            d_inner_product, n_batch
        );    
        
        cudaDeviceSynchronize(); 
        // COUT_ENDL("finish matrix multiply");

        // print_cuda_2D("inner product", d_inner_product, n_query, n_batch);

        // table_cuda_2D("topk index", d_topk_index, n_query, k);
        // table_cuda_2D("topk cos distance", d_topk_cos_dist, n_query, k);
    }

    {
        CUDATimer timer_compute("Kernel Execution: cos + topk", ENABLE_CUDA_TIMING);

        pgvector::fusion_cos_topk_warpsort::fusion_cos_topk_warpsort(
            d_query_norm, d_data_norm, d_inner_product, d_index,
            n_query, n_batch, k,
            d_topk_cos_dist, d_topk_index,
            true /* select min */
        );

        // table_cuda_2D("topk index", d_topk_index, n_query, k);
        // table_cuda_2D("topk cos distance", d_topk_cos_dist, n_query, k);

        cudaDeviceSynchronize(); 
    }

    // ------------------------------------------------------------------
    // Step: 在GPU上并行统计cluster-query映射
    // ------------------------------------------------------------------
    {
        CUDATimer timer_compute("Kernel Execution: Count cluster-query mapping", ENABLE_CUDA_TIMING);
        
        // 第一步：初始化cluster_query_count为0
        cudaMemset(d_cluster_query_count, 0, n_batch * sizeof(int));
        
        // 第二步：并行统计每个cluster被多少个query使用
        dim3 count_block_dim(k);  // 每个block处理一个query的k个probe
        dim3 count_grid_dim(n_query);  // 每个block处理一个query
        count_cluster_queries_kernel<<<count_grid_dim, count_block_dim>>>(
            d_topk_index,
            d_cluster_query_count,
            n_query,
            k,
            n_batch  // n_total_clusters
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
        
        // 第三步：在CPU上构建CSR格式的offset数组（需要先知道count）
        // 注意：这里需要先同步，然后从GPU读取count到CPU构建offset
        int* cluster_query_count_host = (int*)malloc(n_batch * sizeof(int));
        cudaMemcpy(cluster_query_count_host, d_cluster_query_count, n_batch * sizeof(int), cudaMemcpyDeviceToHost);
        
        // 构建offset数组
        int* cluster_query_offset_host = (int*)malloc((n_batch + 1) * sizeof(int));
        cluster_query_offset_host[0] = 0;
        for (int i = 0; i < n_batch; i++) {
            cluster_query_offset_host[i + 1] = cluster_query_offset_host[i] + cluster_query_count_host[i];
        }
        
        // 将offset数组复制回GPU
        cudaMemcpy(d_cluster_query_offset, cluster_query_offset_host, (n_batch + 1) * sizeof(int), cudaMemcpyHostToDevice);
        
        // 初始化写入位置数组（从offset复制）
        cudaMemcpy(d_cluster_write_pos, cluster_query_offset_host, n_batch * sizeof(int), cudaMemcpyHostToDevice);
        
        int total_entries = cluster_query_offset_host[n_batch];
        free(cluster_query_count_host);
        free(cluster_query_offset_host);
        
        // 第四步：并行构建cluster-query映射的CSR格式数据
        build_cluster_query_mapping_kernel<<<count_grid_dim, count_block_dim>>>(
            d_topk_index,
            d_cluster_query_offset,
            d_cluster_query_data,
            d_cluster_query_probe_indices,
            d_cluster_write_pos,
            n_query,
            k,
            n_batch  // n_total_clusters
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
        
        // 第五步：转换为紧凑格式（每个cluster对应4个query）
        // 计算最大可能的组数量（每个cluster最多ceil(n_query/k)个组，但实际会更少）
        // 保守估计：每个cluster最多ceil(n_query/4)个组，但实际每个cluster的query数通常较少
        // 更准确的估计：基于total_entries，每组4个query
        int max_groups = (total_entries + 3) / 4;  // 向上取整
        
        // 重新初始化组数量计数器
        cudaMemset(d_n_groups, 0, sizeof(int));
        
        // 并行转换每个cluster
        dim3 convert_block_dim(1);  // 每个block一个线程
        dim3 convert_grid_dim(n_batch);  // 每个block处理一个cluster
        convert_to_compact_groups_kernel<<<convert_grid_dim, convert_block_dim>>>(
            d_cluster_query_offset,
            d_cluster_query_data,
            d_cluster_query_probe_indices,
            d_compact_groups,
            d_n_groups,
            n_batch  // n_total_clusters
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    }

    {
        CUDATimer timer_trans2("D2H Data Transfer", ENABLE_CUDA_TIMING);
        // 只复制索引，不复制dist（dist作为局部变量）
        cudaMemcpy(h_topk_index[0], d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost);
        // 注意：cluster-query映射已经在GPU上，如果需要可以在这里复制回host
    }

    {
        CUDATimer timer_manage2("GPU Memory Free", ENABLE_CUDA_TIMING, false);
        cublasDestroy(handle);
        cudaFree(d_query_vectors);
        cudaFree(d_data_vectors);
        cudaFree(d_inner_product);
        cudaFree(d_query_norm);
        cudaFree(d_index);
        cudaFree(d_topk_cos_dist);  // 局部变量，释放
        cudaFree(d_topk_index);
        cudaFree(d_cluster_write_pos);  // 临时数组，释放
        // 销毁CUDA流
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }

    // CHECK_CUDA_ERRORS;
}