#include "fusion_cos_topk.cuh"
#include "../indexed_gemm/indexed_gemm.cuh"
#include "../pch.h"
#include "../unit_tests/common/test_utils.cuh"
#include "../warpsortfilter/warpsort_utils.cuh"
#include "../warpsortfilter/warpsort_topk.cu"
#include <algorithm>
#include <cfloat>
#include <vector>

#define ENABLE_CUDA_TIMING 1
using namespace pgvector::warpsort_utils;
using namespace pgvector::warpsort_topk;

/**
 * Kernel: 初始化输出内存为无效值（FLT_MAX 和 -1）
 */
__global__ static void init_invalid_values_kernel(
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
__global__ static void map_candidate_indices_kernel(
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
 * v5版本：Entry-based线程模型的流式融合余弦距离top-k计算
 * 
 * 核心设计：
 * - 每个 block 处理一个 entry（一个 cluster + 一组 query，8个或4个）
 * - grid维度 = n_entry（只处理有query的cluster，避免空block）
 * - 不需要统计max_queries_per_probe，grid维度就是实际entry数量
 * - 好处：不会涉及不需要的cluster，提高并行性
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
 * @param n_total_clusters cluster数量
 * @param n_probes probe数量
 * @param n_dim 向量维度
 * @param k topk数量
 */
void cuda_cos_topk_warpsort_fine_v5(
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

    float** candidate_dist,
    int** candidate_index,

    int n_query,
    int n_total_clusters,
    int n_probes,
    int n_dim,
    int k
) {
    // 检查输入参数有效性
    if (d_probe_query_offsets == nullptr || d_probe_queries == nullptr) {
        printf("[ERROR] Invalid input pointers: d_probe_query_offsets=%p, d_probe_queries=%p\n", 
               d_probe_query_offsets, d_probe_queries);
        return;
    }
    
    
    // 检查k是否在有效范围内
    if (k > kMaxCapacity) {
        printf("Error: k (%d) exceeds maximum capacity (%d)\n", k, kMaxCapacity);
        return;
    }

    int capacity = 32;

    int* probe_query_offsets_host = nullptr;
    int* probe_queries_host = nullptr;
    int* probe_query_probe_indices_host = nullptr;
    constexpr int kQueriesPerBlock = 8;
    float* d_topk_dist_probe = nullptr;
    int* d_topk_index_probe = nullptr;
    
    // Entry数据结构（GPU内存）
    int* d_entry_cluster_id = nullptr;
    int* d_entry_query_start = nullptr;
    int* d_entry_query_count = nullptr;
    int* d_entry_queries = nullptr;
    int* d_entry_probe_indices = nullptr;
    
    // 配置kernel launch
    // v5 entry-based版本：每个block处理一个entry（一个cluster + 一组query）
    dim3 block(256);  // 8个warp，每个warp 32个线程

    int n_entry = 0;

    {
        CUDATimer timer_init("Init Invalid Values Kernel and Build Entries", ENABLE_CUDA_TIMING);

        // 选择合适的Capacity（必须是2的幂，且 > k）
        while (capacity < k) capacity <<= 1;
        capacity = std::min(capacity, kMaxCapacity);
        
        CHECK_CUDA_ERRORS;
        
        // 缓冲区：按query组织的结果 [n_query][n_probes][k]
        // kernel会直接写入这个格式，不需要重组
        cudaMalloc(&d_topk_dist_probe, n_query * n_probes * k * sizeof(float));
        cudaMalloc(&d_topk_index_probe, n_query * n_probes * k * sizeof(int));
    
        // 初始化输出内存为无效值（FLT_MAX 和 -1）
        dim3 init_block(512);
        int init_grid_size = (n_query * n_probes * k + init_block.x - 1) / init_block.x;
        dim3 init_grid(init_grid_size);
        init_invalid_values_kernel<<<init_grid, init_block>>>(
            d_topk_dist_probe,
            d_topk_index_probe,
            n_query * n_probes * k
        );
        CHECK_CUDA_ERRORS;
        
        // 构建entry数据结构
        // 1. 从GPU复制probe_query数据到CPU
        probe_query_offsets_host = (int*)malloc((n_total_clusters + 1) * sizeof(int));
        cudaMemcpy(probe_query_offsets_host, d_probe_query_offsets, (n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS;
        
        int total_queries = probe_query_offsets_host[n_total_clusters];
        if (total_queries > 0) {
            probe_queries_host = (int*)malloc(total_queries * sizeof(int));
            probe_query_probe_indices_host = (int*)malloc(total_queries * sizeof(int));
            cudaMemcpy(probe_queries_host, d_probe_queries, total_queries * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(probe_query_probe_indices_host, d_probe_query_probe_indices, total_queries * sizeof(int), cudaMemcpyDeviceToHost);
            CHECK_CUDA_ERRORS;
        }
        
        // 2. 构建entry列表（临时存储在CPU）
        std::vector<int> entry_cluster_id;
        std::vector<int> entry_query_start;
        std::vector<int> entry_query_count;
        std::vector<int> entry_queries;
        std::vector<int> entry_probe_indices;
        
        int current_query_offset = 0;
        
        // 遍历所有cluster，对于每个有query的cluster，将其query分组为entry
        for (int cluster_id = 0; cluster_id < n_total_clusters; ++cluster_id) {
            int query_start = probe_query_offsets_host[cluster_id];
            int query_end = probe_query_offsets_host[cluster_id + 1];
            int n_queries = query_end - query_start;
            
            if (n_queries == 0) continue;  // 跳过没有query的cluster
            
            // 将query分组为entry（每组kQueriesPerBlock个）
            for (int batch_start = 0; batch_start < n_queries; batch_start += kQueriesPerBlock) {
                int batch_size = std::min(kQueriesPerBlock, n_queries - batch_start);
                
                entry_cluster_id.push_back(cluster_id);
                entry_query_start.push_back(current_query_offset);
                entry_query_count.push_back(batch_size);
                
                // 添加这个entry的query和probe_indices
                for (int i = 0; i < batch_size; ++i) {
                    int query_idx = query_start + batch_start + i;
                    entry_queries.push_back(probe_queries_host[query_idx]);
                    entry_probe_indices.push_back(probe_query_probe_indices_host[query_idx]);
                }
                
                current_query_offset += batch_size;
                n_entry++;
            }
        }
        
        // 3. 将entry数据复制到GPU
        if (n_entry > 0) {
            cudaMalloc(&d_entry_cluster_id, n_entry * sizeof(int));
            cudaMalloc(&d_entry_query_start, n_entry * sizeof(int));
            cudaMalloc(&d_entry_query_count, n_entry * sizeof(int));
            cudaMalloc(&d_entry_queries, entry_queries.size() * sizeof(int));
            cudaMalloc(&d_entry_probe_indices, entry_probe_indices.size() * sizeof(int));
            
            cudaMemcpy(d_entry_cluster_id, entry_cluster_id.data(), n_entry * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_entry_query_start, entry_query_start.data(), n_entry * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_entry_query_count, entry_query_count.data(), n_entry * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_entry_queries, entry_queries.data(), entry_queries.size() * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_entry_probe_indices, entry_probe_indices.data(), entry_probe_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
            CHECK_CUDA_ERRORS;
        }
    }
    
    {
        CUDATimer timer_kernel("Indexed Inner Product with TopK Kernel (v5 entry-based)", ENABLE_CUDA_TIMING);
        
        if (n_entry == 0) {
            // 没有entry，直接返回（所有结果已经是FLT_MAX和-1）
            goto cleanup;
        }
        
        // 根据capacity选择kernel实例
        if (capacity <= 32) {
            launch_indexed_inner_product_with_topk_kernel_v5_entry_based<64, true, kQueriesPerBlock>(
                block,
                n_dim,
                d_query_group,
                d_cluster_vector,
                d_probe_vector_offset,
                d_probe_vector_count,
                d_entry_cluster_id,
                d_entry_query_start,
                d_entry_query_count,
                d_entry_queries,
                d_entry_probe_indices,
                d_query_norm,
                d_cluster_vector_norm,
                n_entry,
                n_probes,
                k,
                d_topk_dist_probe,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                d_topk_index_probe,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                0
            );
        } else if (capacity <= 64) {
            launch_indexed_inner_product_with_topk_kernel_v5_entry_based<128, true, kQueriesPerBlock>(
                block,
                n_dim,
                d_query_group,
                d_cluster_vector,
                d_probe_vector_offset,
                d_probe_vector_count,
                d_entry_cluster_id,
                d_entry_query_start,
                d_entry_query_count,
                d_entry_queries,
                d_entry_probe_indices,
                d_query_norm,
                d_cluster_vector_norm,
                n_entry,
                n_probes,
                k,
                d_topk_dist_probe,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                d_topk_index_probe,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                0
            );
        } else {
            launch_indexed_inner_product_with_topk_kernel_v5_entry_based<256, true, kQueriesPerBlock>(
                block,
                n_dim,
                d_query_group,
                d_cluster_vector,
                d_probe_vector_offset,
                d_probe_vector_count,
                d_entry_cluster_id,
                d_entry_query_start,
                d_entry_query_count,
                d_entry_queries,
                d_entry_probe_indices,
                d_query_norm,
                d_cluster_vector_norm,
                n_entry,
                n_probes,
                k,
                d_topk_dist_probe,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                d_topk_index_probe,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                0
            );
        }
        CHECK_CUDA_ERRORS;
    }
        
    // kernel已经直接写入按query组织的格式 [n_query][n_probes][k]，无需重组
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

    // 规约：将 [n_query][n_probes][k] 归并为 [n_query][k]
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
        
        // 清除错误状态后调用 select_k
        cudaGetLastError();
        cudaError_t select_err = select_k<float, int>(
            d_topk_dist_probe, n_query, n_probes * k, k,
            d_topk_dist, d_topk_index, true, 0
        );
        
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
    
cleanup:
    // 清理主机内存
    if (probe_query_offsets_host != nullptr) {
        free(probe_query_offsets_host);
    }
    if (probe_queries_host != nullptr) {
        free(probe_queries_host);
    }
    if (probe_query_probe_indices_host != nullptr) {
        free(probe_query_probe_indices_host);
    }
    
    // 清理GPU内存
    if (d_entry_cluster_id != nullptr) {
        cudaFree(d_entry_cluster_id);
    }
    if (d_entry_query_start != nullptr) {
        cudaFree(d_entry_query_start);
    }
    if (d_entry_query_count != nullptr) {
        cudaFree(d_entry_query_count);
    }
    if (d_entry_queries != nullptr) {
        cudaFree(d_entry_queries);
    }
    if (d_entry_probe_indices != nullptr) {
        cudaFree(d_entry_probe_indices);
    }
}

