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
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include "fusion_cos_topk.cuh"
#include "../indexed_gemm/indexed_gemm.cuh"
#include "../unit_tests/common/test_utils.cuh"
#include "pch.h"
#include "warpsortfilter/warpsort_utils.cuh"
#include "warpsortfilter/warpsort.cuh"

#define ENABLE_CUDA_TIMING 0

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

    /* 按照 laneId 访问数据 */
    const T* row_inner_product = d_inner_product + row * len;
    const IdxT* row_index = d_index + row * len;
    for (int i = warp_id * kWarpSize + lane; i < len; i += n_warps * kWarpSize) {
        float data_norm = d_data_norm[i];
        float inner_product = row_inner_product[i];
        IdxT index = row_index[i];  /* 修复：使用正确的索引值 */
        float cos_similarity = inner_product / (query_norm * data_norm);
        float cos_distance = 1.0f - cos_similarity;
        queue.add(cos_distance, index);
    }
    
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

/**
 * 支持可变长度的top-k选择kernel
 * 
 * 这个kernel用于精筛场景，每个query的有效候选数量不同。
 * 每个block处理一个query，使用WarpSortFiltered进行top-k选择。
 * 注意：不对不同的有效候选数量进行特殊处理，所有query使用固定的max_len长度。
 */
template<int Capacity, bool Ascending, typename T, typename IdxT>
__global__ void fusion_cos_topk_warpsort_fine_kernel(
    const T* __restrict__ d_query_norm,
    const T* __restrict__ d_data_norm,
    const T* __restrict__ d_inner_product,
    const IdxT* __restrict__ d_index,
    int n_query,
    int max_len,  // 固定长度，所有query使用相同的候选数量
    int k,
    T* __restrict__ output_vals,
    IdxT* __restrict__ output_idx)
{
    using namespace pgvector::warpsort_utils;
    using namespace pgvector::warpsort;
    
    const int query_idx = blockIdx.x;
    if (query_idx >= n_query) return;
    
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = laneId();
    const int n_warps = blockDim.x / kWarpSize;
    
    // 直接使用固定的max_len，不对不同的有效候选数量进行特殊处理
    int len = max_len;
    
    /* 在一个warp中，建立一个长度为k的队列 */
    /* 注意：队列已经在构造函数中用 dummy 值初始化，即使候选数量为0也能正常工作 */
    WarpSortFiltered<Capacity, Ascending, T, IdxT> queue(k);
    
    float query_norm = d_query_norm[query_idx];
    
    // 边界检查：如果query_norm为0，跳过该query
    if (query_norm < 1e-6f) return;

    /* 按照 laneId 访问数据 */
    const T* row_inner_product = d_inner_product + query_idx * max_len;
    const IdxT* row_index = d_index + query_idx * max_len;
    
    // 关键修复：确保所有线程都同步调用 queue.add()
    // queue.add() 内部的 any() 需要所有线程在同一时刻参与，否则会导致死锁
    // 策略：在循环的每次迭代前同步，确保所有线程一起进入 queue.add()
    
    /* 确保所有线程都执行到这里，开始统一处理 */
    __syncwarp();
    
    // 计算最大迭代次数，确保所有线程都执行相同次数的循环
    int max_iter = (len + n_warps * kWarpSize - 1) / (n_warps * kWarpSize);
    
    for (int iter = 0; iter < max_iter; iter++) {
        int i = warp_id * kWarpSize + lane + iter * n_warps * kWarpSize;
        
        // 确保所有线程在每个迭代中都同步
        __syncwarp();
        
        bool has_data = (i < len);
        float inner_product = 0.0f;
        IdxT idx = (IdxT)-1;
        
        if (has_data) {
            inner_product = row_inner_product[i];
            idx = row_index[i];
            
            // 跳过无效的索引（-1表示未使用）
            if (idx < 0) {
                has_data = false;
            } else {
                // 使用索引获取对应的向量norm
                float data_norm = d_data_norm[idx];
                
                // 边界检查：避免除零
                if (data_norm < 1e-6f) {
                    has_data = false;
                }
            }
        }
        
        // 再次同步，确保所有线程都完成了数据加载和检查
        __syncwarp();
        
        // 现在所有线程统一调用 queue.add()
        if (has_data) {
            float cos_similarity = inner_product / (query_norm * d_data_norm[idx]);
            float cos_distance = 1.0f - cos_similarity;
            queue.add(cos_distance, idx);
        } 
        else {
            // 没有有效数据的线程添加 dummy 值
            T dummy_val = Ascending ? upper_bound<T>() : lower_bound<T>();
            queue.add(dummy_val, (IdxT)-1);
        }
    }
    
    /* 确保所有线程都完成了所有迭代 */
    __syncwarp();
    
    /* 把 buffer 中剩余数合并到 queue 中 */
    queue.done();
    
    /* 再次同步，确保 done() 完成 */
    __syncwarp();
    
    /* 将 queue 中的数存储到显存中（所有线程都要调用）*/
    if (warp_id == 0) {
        T* row_out_val = output_vals + query_idx * k;
        IdxT* row_out_idx = output_idx + query_idx * k;
        queue.store(row_out_val, row_out_idx);
    }
}

} // namespace fusion_cos_topk_warpsort
} // namespace pgvector

// ============================================================================
// Fine Screening Interface
// ============================================================================

/**
 * 适配fine_screen_top_n的融合余弦距离top-k计算
 * 
 * 这个函数专门用于精筛场景，其中query不是连续存储的，
 * 而是通过cluster_query_offset和cluster_query_data来索引。
 * 向量按聚类物理连续存储（符合pgvector实现）。
 * 
 * 使用优化的内存分配策略（参考cuVS）：
 * - 在host端预先计算每个query的实际候选数
 * - 使用max_candidates_per_query（实际最大值）而非固定上限分配内存
 * - 使用d_num_samples数组在kernel中限制处理范围
 */
void cuda_cos_topk_warpsort_fine(
    const float* d_query_group,         /* query向量数据 */
    const float* d_cluster_vector,
    const int* d_cluster_query_offset,
    const int* d_cluster_query_data,
    const int* d_cluster_vector_index,  // 每个cluster在全局向量数组中的连续起始位置
    const int* d_cluster_vector_num,
    const float* d_query_norm,
    const float* d_cluster_vector_norm,
    int* d_topk_index,                  /* 结果：topk 索引 */
    float* d_topk_dist,                 /* 结果：topk 剧烈 */
    int n_query,                        /* batch中的query数量 */
    int n_total_cluster,                /* 聚类总数 */
    int n_dim,
    int k,                              /* topk的数量 */
    int n_total_vectors,                /**/
    int max_candidates_per_query,       // 所有query的最大候选数（在host端计算，基于实际值）
    const int* d_num_samples            // 每个query的实际候选数量 [n_query]（用于限制处理范围）
) {
    /**
     * 策略：
     * 1. 为每个cluster，计算对应的query与cluster向量的内积
     * 2. 对于每个query，收集所有相关的内积和索引
     * 3. 对每个query进行top-k选择
     */
    
    // 使用优化后的内存分配：基于实际计算的最大候选数，而不是固定上限
    // max_candidates_per_query已经在host端计算，是实际的最大值
    size_t size_inner_product = n_query * max_candidates_per_query * sizeof(float);
    size_t size_index = n_query * max_candidates_per_query * sizeof(int);
    size_t size_query_pos_atomic = n_query * sizeof(int);  // 原子计数器数组，用于分配存储位置
    
    float* d_inner_product;
    int* d_index;
    int* d_query_pos_atomic;  // 每个query的位置分配原子计数器（不跟踪数量）
    
    {
        CUDATimer timer_alloc("Memory Allocation", ENABLE_CUDA_TIMING);
        cudaMalloc(&d_inner_product, size_inner_product);
        cudaMalloc(&d_index, size_index);
        cudaMalloc(&d_query_pos_atomic, size_query_pos_atomic);
        cudaMemset(d_query_pos_atomic, 0, size_query_pos_atomic);  // 初始化为0

        // // 初始化
        // thrust::fill(
        //     thrust::device_pointer_cast(d_inner_product),
        //     thrust::device_pointer_cast(d_inner_product) + (n_query * max_candidates_per_query),
        //     0.0f
        // );
        // thrust::fill(
        //     thrust::device_pointer_cast(d_index),
        //     thrust::device_pointer_cast(d_index) + (n_query * max_candidates_per_query),
        //     -1
        // );
        // cudaMemset(d_query_count, 0, size_query_count);
    }
    
    {
        CUDATimer timer_inner_product("Indexed Inner Product Kernel", ENABLE_CUDA_TIMING);
        // 调用索引化的内积计算kernel
        // 每个block处理一个cluster
        dim3 grid(n_total_cluster);
        dim3 block(256); // 每个block使用256个线程
        
        // 暂时禁用共享内存优化，避免misaligned address错误
        const size_t smem_size = 0;
        
        indexed_inner_product_kernel<<<grid, block, smem_size>>>(
            d_query_group,
            d_cluster_vector,
            d_cluster_query_offset,
            d_cluster_query_data,
            d_cluster_vector_index,  // 连续起始位置数组
            d_cluster_vector_num,
            d_inner_product,
            d_index,
            d_query_pos_atomic,  // 原子计数器数组，仅用于分配存储位置
            d_num_samples,  // 每个query的实际候选数量（用于边界检查）
            n_query,
            n_total_cluster,
            n_dim,
            n_total_vectors,
            max_candidates_per_query
        );
        
        cudaDeviceSynchronize();
    }
    
    // Debug: 检查中间结果（仅在小规模测试时）
    // 注意：不再跟踪每个query的有效候选数量，所有query使用固定的长度
    if (!QUIET) {
        float* h_inner_product_sample = (float*)malloc(n_query * 4 * sizeof(float));
        int* h_index_sample = (int*)malloc(n_query * 4 * sizeof(int));
        
        cudaMemcpy(h_inner_product_sample, d_inner_product, n_query * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_index_sample, d_index, n_query * 4 * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("\n=== GPU Debug: Intermediate Results ===\n");
        printf("d_inner_product (first query, first 4): ");
        for (int i = 0; i < 4; i++) {
            printf("%.4f ", h_inner_product_sample[i]);
        }
        printf("\n");
        printf("d_index (first query, first 4): ");
        for (int i = 0; i < 4; i++) {
            printf("%d ", h_index_sample[i]);
        }
        printf("\n");
        
        free(h_inner_product_sample);
        free(h_index_sample);
    }
    
    // 为每个query调用top-k选择kernel
    // 选择capacity（在块外定义，以便debug代码访问）
    int capacity = 32;
    while (capacity < k) capacity <<= 1;
    capacity = min(capacity, 256);  // 限制最大capacity
    
    dim3 topk_grid(n_query);
    dim3 topk_block(32);
    
    {
        CUDATimer timer_topk("Top-K Selection Kernel", ENABLE_CUDA_TIMING);
        
        // 调用top-k选择kernel，所有query使用固定的候选数量
        if (capacity <= 32) {
            pgvector::fusion_cos_topk_warpsort::fusion_cos_topk_warpsort_fine_kernel<64, true, float, int><<<topk_grid, topk_block>>>(
                d_query_norm,
                d_cluster_vector_norm,
                d_inner_product,
                d_index,
                n_query,
                max_candidates_per_query,  // 使用实际的最大候选数（而非固定上限）
                k,
                d_topk_dist,
                d_topk_index
            );
        } else if (capacity <= 64) {
            pgvector::fusion_cos_topk_warpsort::fusion_cos_topk_warpsort_fine_kernel<128, true, float, int><<<topk_grid, topk_block>>>(
                d_query_norm,
                d_cluster_vector_norm,
                d_inner_product,
                d_index,
                n_query,
                max_candidates_per_query,  // 使用实际的最大候选数（而非固定上限）
                k,
                d_topk_dist,
                d_topk_index
            );
        } else {
            pgvector::fusion_cos_topk_warpsort::fusion_cos_topk_warpsort_fine_kernel<256, true, float, int><<<topk_grid, topk_block>>>(
                d_query_norm,
                d_cluster_vector_norm,
                d_inner_product,
                d_index,
                n_query,
                max_candidates_per_query,  // 使用实际的最大候选数（而非固定上限）
                k,
                d_topk_dist,
                d_topk_index
            );
        }
        
        // 检查 kernel launch 错误
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess && true) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(launch_err));
        }
        
        if (!QUIET) {
            printf("About to synchronize after top-k kernel...\n");
            fflush(stdout);
        }
        
        cudaDeviceSynchronize();
    }
    
    // Debug: 检查top-k结果
    if (!QUIET) {
        printf("\n=== GPU Debug: Top-K Kernel Info ===\n");
        printf("k=%d, capacity=%d, grid=(%d,1,1), block=(%d,1,1)\n", 
               k, capacity, n_query, 32);
        
        float* h_topk_dist = (float*)malloc(n_query * k * sizeof(float));
        int* h_topk_index = (int*)malloc(n_query * k * sizeof(int));
        
        cudaMemcpy(h_topk_dist, d_topk_dist, n_query * k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_topk_index, d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("d_topk_dist:\n");
        for (int i = 0; i < n_query; i++) {
            for (int j = 0; j < k; j++) {
                printf("%.4f ", h_topk_dist[i * k + j]);
            }
            printf("\n");
        }
        printf("d_topk_index:\n");
        for (int i = 0; i < n_query; i++) {
            for (int j = 0; j < k; j++) {
                printf("%d ", h_topk_index[i * k + j]);
            }
            printf("\n");
        }
        
        free(h_topk_dist);
        free(h_topk_index);
    }
    
    {
        CUDATimer timer_free("Memory Free", ENABLE_CUDA_TIMING);
        // 清理临时内存
        cudaFree(d_inner_product);
        cudaFree(d_index);
        cudaFree(d_query_pos_atomic);
    }
}
