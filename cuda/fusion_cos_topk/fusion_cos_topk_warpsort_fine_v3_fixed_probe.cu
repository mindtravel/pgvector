#include "fusion_cos_topk.cuh"
#include "../indexed_gemm/indexed_gemm.cuh"
#include "../pch.h"
#include "../unit_tests/common/test_utils.cuh"
#include "../warpsortfilter/warpsort_utils.cuh"
#include "../warpsortfilter/warpsort_topk.cu"
#include <algorithm>
#include <cfloat>

#define ENABLE_CUDA_TIMING 0
using namespace pgvector::warpsort_utils;
using namespace pgvector::warpsort_topk;

/**
 * 辅助函数：在CPU上输出精筛候选结果（用于调试）
 */
void print_fine_search_candidates_cpu(
    const float* d_topk_dist_probe,
    const int* d_topk_index_probe,
    int n_query,
    int n_probes,
    int k
) {
    printf("\n=== Fine Search Candidates (Before Reduction) ===\n");
    printf("Parameters: n_query=%d, n_probes=%d, k=%d\n", n_query, n_probes, k);
    fflush(stdout);
    
    // 限制输出规模，避免输出过多
    if (n_query > 10 || n_probes * k > 1000) {
        printf("(Skipping detailed output due to large size: n_query=%d, n_probes*k=%d)\n", 
               n_query, n_probes * k);
        fflush(stdout);
        return;
    }
    int max_candidates_per_query = n_probes * k;
    
    // 将数据从GPU复制到CPU
    float* h_candidate_dists = (float*)malloc(n_query * max_candidates_per_query * sizeof(float));
    int* h_candidate_indices = (int*)malloc(n_query * max_candidates_per_query * sizeof(int));
    
    cudaMemcpy(h_candidate_dists, d_topk_dist_probe, 
               n_query * max_candidates_per_query * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_candidate_indices, d_topk_index_probe, 
               n_query * max_candidates_per_query * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;
    
    // 在CPU上输出
    for (int qi = 0; qi < n_query; ++qi) {
        printf("Query %d (has %d probes):\n", qi, n_probes);
        int valid_count = 0;
        for (int ci = 0; ci < max_candidates_per_query; ++ci) {
            int idx = qi * max_candidates_per_query + ci;
            float dist = h_candidate_dists[idx];
            int vec_idx = h_candidate_indices[idx];
            
            // 检查是否是有效值（不是INF和-1）
            if (dist < FLT_MAX && vec_idx >= 0) {
                printf("  candidate[%d]: idx=%d, dist=%.6f\n", ci, vec_idx, dist);
                valid_count++;
                if (valid_count >= 20) {  // 限制输出数量
                    printf("  ... (showing first 20 candidates)\n");
                    break;
                }
            }
        }
        if (valid_count == 0) {
            printf("  (no valid candidates)\n");
        }
    }
    
    free(h_candidate_dists);
    free(h_candidate_indices);
}

/**
 * Kernel: 初始化输出内存为无效值（FLT_MAX 和 -1）
 */
__global__ void init_invalid_values_kernel(
    float* __restrict__ d_topk_dist_probe,  // [n_query][n_probes][k] - 输出，初始化为 FLT_MAX
    int* __restrict__ d_topk_index_probe,  // [n_query][n_probes][k] - 输出，初始化为 -1
    int total_size  // n_query * n_probes * k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    d_topk_dist_probe[idx] = FLT_MAX;
    d_topk_index_probe[idx] = -1;
}

/**
 * Kernel: 映射候选索引回原始向量索引
 */
__global__ void map_candidate_indices_kernel(
    const int* __restrict__ d_candidate_indices,  // [n_query][n_probes * k]
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
 * 固定 probe 版本的流式融合余弦距离top-k计算
 * 
 * 核心设计：
 * - 每个 block 处理一个 probe 的多个 query
 * - gridDim.x = n_probes（每个 block 一个 probe）
 * - gridDim.y = query batch 数量（每个 block 处理一个 probe 的一个 query batch）
 * - 利用 L2 cache：多个 query 访问相同的 probe 向量数据
 * 
 * @param d_query_group query向量 [n_query * n_dim]
 * @param d_cluster_vector 所有向量数据（连续存储）[n_total_vectors * n_dim]
 * @param d_probe_vector_offset 每个probe在d_cluster_vector中的起始位置 [n_probes]
 * @param d_probe_vector_count 每个probe的向量数量 [n_probes]
 * @param d_probe_queries probe对应的query列表（CSR格式）[total_queries]
 * @param d_probe_query_offsets probe的query列表起始位置（CSR格式）[n_probes + 1]
 * @param d_query_norm query的l2norm [n_query]
 * @param d_cluster_vector_norm 所有向量的l2norm [n_total_vectors]
 * @param d_probe_query_probe_indices 每个probe-query对中probe在query中的索引 [total_queries_in_probes]
 * @param d_topk_index [out] 每个query的topk索引 [n_query][k]（最终结果，已规约）
 * @param d_topk_dist [out] 每个query的topk距离 [n_query][k]（最终结果，已规约）
 * @param n_query query数量
 * @param n_probes probe数量
 * @param n_dim 向量维度
 * @param k topk数量
 */
void cuda_cos_topk_warpsort_fine_v3_fixed_probe(
    float* d_query_group,
    float* d_cluster_vector,

    int* d_probe_vector_offset,
    int* d_probe_vector_count,
    int* d_probe_queries,
    int* d_probe_query_offsets,
    int* d_probe_query_probe_indices,

    float* d_query_norm,
    float* d_cluster_vector_norm,

    int* d_topk_index,
    float* d_topk_dist,

    float** candidate_index,
    int** candidate_dist,

    int n_query,
    int n_probes,
    int n_dim,
    int k
) {
    printf("[DEBUG] cuda_cos_topk_warpsort_fine_v3_fixed_probe ENTER: n_query=%d, n_probes=%d, k=%d\n", 
           n_query, n_probes, k);
    printf("[DEBUG] Pointers: d_probe_query_offsets=%p, d_probe_queries=%p\n", 
           d_probe_query_offsets, d_probe_queries);
    fflush(stdout);
    
    // 检查输入参数有效性
    if (d_probe_query_offsets == nullptr || d_probe_queries == nullptr) {
        printf("[ERROR] Invalid input pointers: d_probe_query_offsets=%p, d_probe_queries=%p\n", 
               d_probe_query_offsets, d_probe_queries);
        return;
    }
    
    // 验证指针是否指向有效的设备内存
    cudaPointerAttributes attr_offsets, attr_queries;
    cudaError_t err1 = cudaPointerGetAttributes(&attr_offsets, d_probe_query_offsets);
    cudaError_t err2 = cudaPointerGetAttributes(&attr_queries, d_probe_queries);
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        printf("[ERROR] Failed to get pointer attributes: err1=%s, err2=%s\n", 
               cudaGetErrorString(err1), cudaGetErrorString(err2));
        return;
    }
    printf("[DEBUG] Pointer attributes: offsets.type=%d (0=Host, 2=Device), queries.type=%d\n", 
           attr_offsets.type, attr_queries.type);
    if (attr_offsets.type != cudaMemoryTypeDevice || attr_queries.type != cudaMemoryTypeDevice) {
        printf("[ERROR] Pointers are not device memory: offsets.type=%d, queries.type=%d\n", 
               attr_offsets.type, attr_queries.type);
        printf("[ERROR] Expected cudaMemoryTypeDevice=%d\n", cudaMemoryTypeDevice);
        return;
    }
    
    // 检查k是否在有效范围内
    if (k > kMaxCapacity) {
        printf("Error: k (%d) exceeds maximum capacity (%d)\n", k, kMaxCapacity);
        return;
    }
    
    // 选择合适的Capacity（必须是2的幂，且 > k）
    int capacity = 32;
    while (capacity < k) capacity <<= 1;
    capacity = std::min(capacity, kMaxCapacity);
    
    CHECK_CUDA_ERRORS;
    
    // 配置kernel launch
    // 固定probe版本：每个block处理一个probe的多个query
    constexpr int kQueriesPerBlock = 8;
    dim3 block(256);  // 8个warp，每个warp 32个线程
    
    // 计算max_queries_per_probe用于launch函数（用于计算grid配置）
    int* probe_query_offsets_host = (int*)malloc((n_probes + 1) * sizeof(int));
    cudaMemcpy(probe_query_offsets_host, d_probe_query_offsets, (n_probes + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    int max_queries_per_probe = 0;
    for (int i = 0; i < n_probes; ++i) {
        int n_queries = probe_query_offsets_host[i + 1] - probe_query_offsets_host[i];
        max_queries_per_probe = std::max(max_queries_per_probe, n_queries);
    }
    CHECK_CUDA_ERRORS;

    float* d_topk_dist_probe = nullptr;
    int* d_topk_index_probe = nullptr;
    int total_size = n_query * n_probes * k;
    
    // 检查整数溢出
    if (total_size < 0 || total_size / n_query != n_probes * k) {
        printf("[ERROR] Integer overflow in total_size calculation: n_query=%d, n_probes=%d, k=%d\n",
               n_query, n_probes, k);
        return;
    }
    
    cudaError_t alloc_err1 = cudaMalloc(&d_topk_dist_probe, total_size * sizeof(float));
    cudaError_t alloc_err2 = cudaMalloc(&d_topk_index_probe, total_size * sizeof(int));
    
    if (alloc_err1 != cudaSuccess || alloc_err2 != cudaSuccess) {
        printf("[ERROR] Failed to allocate device memory: dist_err=%s, idx_err=%s\n",
               cudaGetErrorString(alloc_err1), cudaGetErrorString(alloc_err2));
        if (d_topk_dist_probe) cudaFree(d_topk_dist_probe);
        if (d_topk_index_probe) cudaFree(d_topk_index_probe);
        return;
    }
    
    // 初始化输出内存为无效值（FLT_MAX 和 -1）
    dim3 init_block(256);
    int init_grid_size = (total_size + init_block.x - 1) / init_block.x;
    
    // 检查 grid 大小是否超过 CUDA 限制（65535）
    if (init_grid_size > 65535) {
        printf("[ERROR] Grid size too large: %d (max 65535), total_size=%d\n", 
               init_grid_size, total_size);
        cudaFree(d_topk_dist_probe);
        cudaFree(d_topk_index_probe);
        return;
    }
    
    dim3 init_grid(init_grid_size);
    init_invalid_values_kernel<<<init_grid, init_block>>>(
        d_topk_dist_probe,
        d_topk_index_probe,
        total_size
    );
    
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("[ERROR] init_invalid_values_kernel launch failed: %s\n", 
               cudaGetErrorString(launch_err));
        cudaFree(d_topk_dist_probe);
        cudaFree(d_topk_index_probe);
        return;
    }
    
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        printf("[ERROR] init_invalid_values_kernel execution failed: %s\n", 
               cudaGetErrorString(sync_err));
        cudaFree(d_topk_dist_probe);
        cudaFree(d_topk_index_probe);
        return;
    }
    
    CHECK_CUDA_ERRORS;
    
    {
        CUDATimer timer_kernel("Indexed Inner Product with TopK Kernel (v3 fixed probe)", ENABLE_CUDA_TIMING);
        
        // 验证 grid 配置
        int max_query_batches = (max_queries_per_probe + kQueriesPerBlock - 1) / kQueriesPerBlock;
        dim3 grid(n_probes, max_query_batches, 1);
        
        // 检查 grid 大小是否超过 CUDA 限制
        if (grid.x > 2147483647 || grid.y > 65535 || grid.z > 65535) {
            printf("[ERROR] Grid size too large: grid=(%d, %d, %d), n_probes=%d, max_queries_per_probe=%d, max_query_batches=%d\n",
                   grid.x, grid.y, grid.z, n_probes, max_queries_per_probe, max_query_batches);
            cudaFree(d_topk_dist_probe);
            cudaFree(d_topk_index_probe);
            return;
        }
        
        // 根据capacity选择kernel实例
        if (capacity <= 32) {
            launch_indexed_inner_product_with_topk_kernel_v3_fixed_probe<64, true, kQueriesPerBlock>(
                block,
                n_dim,
                d_query_group,
                d_cluster_vector,
                d_probe_vector_offset,
                d_probe_vector_count,
                d_probe_queries,
                d_probe_query_offsets,
                d_probe_query_probe_indices,
                d_query_norm,
                d_cluster_vector_norm,
                n_probes,
                max_queries_per_probe,
                k,
                d_topk_dist_probe,  // 中间缓冲区
                d_topk_index_probe,  // 中间缓冲区
                0
            );
        } else if (capacity <= 64) {
            launch_indexed_inner_product_with_topk_kernel_v3_fixed_probe<128, true, kQueriesPerBlock>(
                block,
                n_dim,
                d_query_group,
                d_cluster_vector,
                d_probe_vector_offset,
                d_probe_vector_count,
                d_probe_queries,
                d_probe_query_offsets,
                d_probe_query_probe_indices,
                d_query_norm,
                d_cluster_vector_norm,
                n_probes,
                max_queries_per_probe,
                k,
                d_topk_dist_probe,  // 中间缓冲区
                d_topk_index_probe,  // 中间缓冲区
                0
            );
        } else {
            launch_indexed_inner_product_with_topk_kernel_v3_fixed_probe<256, true, kQueriesPerBlock>(
                block,
                n_dim,
                d_query_group,
                d_cluster_vector,
                d_probe_vector_offset,
                d_probe_vector_count,
                d_probe_queries,
                d_probe_query_offsets,
                d_probe_query_probe_indices,
                d_query_norm,
                d_cluster_vector_norm,
                n_probes,
                max_queries_per_probe,
                k,
                d_topk_dist_probe,  // 中间缓冲区
                d_topk_index_probe,  // 中间缓冲区
                0
            );
        }
        
        // 检查 kernel 启动错误
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            printf("[ERROR] Fine search kernel launch failed: %s\n", 
                   cudaGetErrorString(launch_err));
            cudaFree(d_topk_dist_probe);
            cudaFree(d_topk_index_probe);
            return;
        }
    }
    
    // 将精筛结果复制到CPU（用于调试输出）
    // candidate_index 和 candidate_dist 是连续内存，[0]指向连续内存的开始
    // 先同步，确保之前的kernel执行完成
    sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        printf("[ERROR] Fine search kernel execution failed: %s\n", 
               cudaGetErrorString(sync_err));
        cudaFree(d_topk_dist_probe);
        cudaFree(d_topk_index_probe);
        return;
    }
    
    // 清除错误状态
    cudaGetLastError();

    if (candidate_index != nullptr && candidate_dist != nullptr && 
        candidate_index[0] != nullptr && candidate_dist[0] != nullptr) {
        cudaMemcpy(candidate_index[0], 
                  d_topk_index_probe, 
                  n_query * n_probes * k * sizeof(int), 
                  cudaMemcpyDeviceToHost);
        cudaMemcpy(candidate_dist[0], 
                  d_topk_dist_probe, 
                  n_query * n_probes * k * sizeof(float), 
                  cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS;
    }

    // 规约：将 [n_query][n_probes_per_query][k] 归并为 [n_query][k]
    // 在GPU上完成，避免CPU-GPU数据复制
    {
        CUDATimer timer_reduce("Reduce probe results to query top-k", ENABLE_CUDA_TIMING);
        
        // 先同步，确保之前的kernel执行完成
        cudaError_t sync_before = cudaDeviceSynchronize();
        if (sync_before != cudaSuccess) {
            printf("[ERROR] Synchronization failed before reduction: %s\n", 
                   cudaGetErrorString(sync_before));
            cudaFree(d_topk_dist_probe);
            cudaFree(d_topk_index_probe);
            return;
        }
        
        // 清除任何残留的错误状态
        cudaGetLastError();
        
        // 验证输入指针有效性（使用 cudaPointerGetAttributes）
        cudaPointerAttributes attr_input_dist, attr_input_idx, attr_output_dist, attr_output_idx;
        cudaError_t err1 = cudaPointerGetAttributes(&attr_input_dist, d_topk_dist_probe);
        cudaError_t err2 = cudaPointerGetAttributes(&attr_input_idx, d_topk_index_probe);
        cudaError_t err3 = cudaPointerGetAttributes(&attr_output_dist, d_topk_dist);
        cudaError_t err4 = cudaPointerGetAttributes(&attr_output_idx, d_topk_index);
        
        if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess) {
            printf("[ERROR] Pointer validation failed: err1=%s, err2=%s, err3=%s, err4=%s\n",
                   cudaGetErrorString(err1), cudaGetErrorString(err2), 
                   cudaGetErrorString(err3), cudaGetErrorString(err4));
            cudaFree(d_topk_dist_probe);
            cudaFree(d_topk_index_probe);
            return;
        }
        
        if (attr_input_dist.type != cudaMemoryTypeDevice || attr_input_idx.type != cudaMemoryTypeDevice ||
            attr_output_dist.type != cudaMemoryTypeDevice || attr_output_idx.type != cudaMemoryTypeDevice) {
            printf("[ERROR] Pointers not device memory: input_dist.type=%d, input_idx.type=%d, output_dist.type=%d, output_idx.type=%d\n",
                   attr_input_dist.type, attr_input_idx.type, attr_output_dist.type, attr_output_idx.type);
            cudaFree(d_topk_dist_probe);
            cudaFree(d_topk_index_probe);
            return;
        }
        
        // 1. 使用select_k并行规约所有query（直接使用原数组，已初始化为FLT_MAX和-1）
        // 验证输出指针有效性
        if (d_topk_dist == nullptr || d_topk_index == nullptr) {
            printf("[ERROR] Output pointers are null: d_topk_dist=%p, d_topk_index=%p\n",
                   d_topk_dist, d_topk_index);
            cudaFree(d_topk_dist_probe);
            cudaFree(d_topk_index_probe);
            return;
        }
        
        // 验证输入指针有效性
        if (d_topk_dist_probe == nullptr || d_topk_index_probe == nullptr) {
            printf("[ERROR] Input pointers are null: d_topk_dist_probe=%p, d_topk_index_probe=%p\n",
                   d_topk_dist_probe, d_topk_index_probe);
            cudaFree(d_topk_dist_probe);
            cudaFree(d_topk_index_probe);
            return;
        }
        
        // 清除错误状态后调用 select_k
        cudaGetLastError();
        cudaError_t select_err = select_k<float, int>(
            d_topk_dist_probe, n_query, n_probes * k, k,
            d_topk_dist, d_topk_index, true, 0
        );
        
        // 同步并检查执行错误
        cudaError_t sync_after = cudaDeviceSynchronize();
        cudaError_t exec_err = cudaGetLastError();
        
        if (select_err != cudaSuccess) {
            printf("[ERROR] select_k launch failed: %s (n_query=%d, n_probes=%d, k=%d, n_probes * k=%d)\n",
                   cudaGetErrorString(select_err), n_query, n_probes, k, n_probes * k);
            printf("[ERROR] Pointers: input_dist=%p, input_idx=%p, output_dist=%p, output_idx=%p\n",
                   d_topk_dist_probe, d_topk_index_probe, d_topk_dist, d_topk_index);
            cudaFree(d_topk_dist_probe);
            cudaFree(d_topk_index_probe);
            return;
        }
        
        if (sync_after != cudaSuccess || exec_err != cudaSuccess) {
            printf("[ERROR] select_k execution failed: sync_err=%s, exec_err=%s (n_query=%d, n_probes=%d, k=%d, n_probes * k=%d)\n",
                   cudaGetErrorString(sync_after), cudaGetErrorString(exec_err), n_query, n_probes, k, n_probes * k);
            printf("[ERROR] Pointers: input_dist=%p, input_idx=%p, output_dist=%p, output_idx=%p\n",
                   d_topk_dist_probe, d_topk_index_probe, d_topk_dist, d_topk_index);
            cudaFree(d_topk_dist_probe);
            cudaFree(d_topk_index_probe);
            return;
        }
        
        // 2. 映射回原始向量索引
        // select_k返回的索引是候选数组中的位置，需要映射回原始向量索引
        dim3 map_block(256);
        dim3 map_grid((n_query * k + map_block.x - 1) / map_block.x);
        map_candidate_indices_kernel<<<map_grid, map_block>>>(
            d_topk_index_probe,  // 使用原数组作为候选索引
            d_topk_index,
            n_query,
            n_probes,
            k
        );
        CHECK_CUDA_ERRORS;
        
        // 清理临时内存
        cudaFree(d_topk_dist_probe);
        cudaFree(d_topk_index_probe);
    }
}

