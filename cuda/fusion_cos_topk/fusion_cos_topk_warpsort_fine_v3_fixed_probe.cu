#include "fusion_cos_topk.cuh"
#include "../indexed_gemm/indexed_gemm.cuh"
#include "../pch.h"
#include "../unit_tests/common/test_utils.cuh"
#include "../utils.cuh"
#include "../warpsortfilter/warpsort_utils.cuh"
#include "../warpsortfilter/warpsort_topk.cu"
#include <algorithm>
#include <cfloat>

#define ENABLE_CUDA_TIMING 1
using namespace pgvector::warpsort_utils;
using namespace pgvector::warpsort_topk;

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
 * @param n_total_clusters cluster数量
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
    constexpr int kQueriesPerBlock = 8;
    float* d_topk_dist_probe = nullptr;
    int* d_topk_index_probe = nullptr;
    
    // 配置kernel launch
    // 固定probe版本：每个block处理一个probe的多个query
    dim3 block(256);  // 8个warp，每个warp 32个线程

    int max_queries_per_probe = 0;

    {
        CUDATimer timer_init("Init Invalid Values Kernel", ENABLE_CUDA_TIMING);

        // 选择合适的Capacity（必须是2的幂，且 > k）
        while (capacity < k) capacity <<= 1;
        capacity = std::min(capacity, kMaxCapacity);
        
        CHECK_CUDA_ERRORS;
        

        
        // 计算max_queries_per_probe用于launch函数（用于计算grid配置）
        // 注意：d_probe_query_offsets 的大小是 n_total_clusters + 1，因为每个 cluster 都是一个 probe
        probe_query_offsets_host = (int*)malloc((n_total_clusters + 1) * sizeof(int));
        cudaMemcpy(probe_query_offsets_host, d_probe_query_offsets, (n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < n_total_clusters; ++i) {
            int n_queries = probe_query_offsets_host[i + 1] - probe_query_offsets_host[i];
            max_queries_per_probe = std::max(max_queries_per_probe, n_queries);
        }
        CHECK_CUDA_ERRORS;

        // 缓冲区：按query组织的结果 [n_query][n_probes][k]
        // kernel会直接写入这个格式，不需要重组
        cudaMalloc(&d_topk_dist_probe, n_query * n_probes * k * sizeof(float));
        cudaMalloc(&d_topk_index_probe, n_query * n_probes * k * sizeof(int));
    
        // 初始化输出内存为无效值（FLT_MAX 和 -1）
        dim3 init_block(512);
        int init_grid_size = (n_query * n_probes * k + init_block.x - 1) / init_block.x;
        
        // // 检查 grid 大小是否超过 CUDA 限制（65535）
        // if (init_grid_size > 65535) {
        //     printf("[ERROR] Grid size too large: %d (max 65535), total_size=%d\n", 
        //            init_grid_size, total_size);
        //     cudaFree(d_topk_dist_probe);
        //     cudaFree(d_topk_index_probe);
        //     return;
        // }
        
        dim3 init_grid(init_grid_size);
        init_invalid_values_kernel<<<init_grid, init_block>>>(
            d_topk_dist_probe,
            d_topk_index_probe,
            n_query * n_probes * k
        );
    }
    int max_query_batches = 0;
    {
        CUDATimer timer_kernel("Indexed Inner Product with TopK Kernel (v3 fixed probe)", ENABLE_CUDA_TIMING);
        
        // 验证 grid 配置
        // grid.x 使用 n_total_clusters，因为每个 cluster 都是一个 probe
        max_query_batches = (max_queries_per_probe + kQueriesPerBlock - 1) / kQueriesPerBlock;
        dim3 grid(n_total_clusters, max_query_batches, 1);
        
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
                n_total_clusters,  // 传递总的 cluster 数量（用于grid.x和检查probe_id）
                n_probes,  // 传递每个query的probe数量（用于检查probe_index_in_query和计算输出位置）
                max_queries_per_probe,
                k,
                d_topk_dist_probe,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                d_topk_index_probe,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
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
                n_total_clusters,  // 传递总的 cluster 数量（用于grid.x和检查probe_id）
                n_probes,  // 传递每个query的probe数量（用于检查probe_index_in_query和计算输出位置）
                max_queries_per_probe,
                k,
                d_topk_dist_probe,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                d_topk_index_probe,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
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
                n_total_clusters,  // 传递总的 cluster 数量（用于grid.x和检查probe_id）
                n_probes,  // 传递每个query的probe数量（用于检查probe_index_in_query和计算输出位置）
                max_queries_per_probe,
                k,
                d_topk_dist_probe,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                d_topk_index_probe,  // 直接写入按query组织的缓冲区 [n_query][n_probes][k]
                0
            );
        }
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
    
    // 清理主机内存
    if (probe_query_offsets_host != nullptr) {
        free(probe_query_offsets_host);
    }
}

