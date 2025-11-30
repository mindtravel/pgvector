#include "fusion_cos_topk.cuh"
#include "../indexed_gemm/indexed_gemm.cuh"
#include "../pch.h"
#include "../unit_tests/common/test_utils.cuh"
#include "../warpsortfilter/warpsort_utils.cuh"
#include <algorithm>

#define ENABLE_CUDA_TIMING 1
using namespace pgvector::warpsort_utils;

/**
 * 流式融合余弦距离top-k计算（v1版本）
 * 
 * 这个函数使用流式计算方式，在kernel内部维护warp-sort queue，直接写入最终结果。
 * 内存占用从 O(n_query * max_candidates) 降至 O(n_query * k)。
 * 
 * 关键改进：
 * 1. 移除了中间缓冲区 d_inner_product 和 d_index
 * 2. 在 indexed_inner_product_with_topk_kernel 中直接维护 topk
 * 3. 直接写入最终输出 [n_query, k]
 * 
 * 限制：
 * - k <= 256 (warp-sort容量限制)
 * - 当前实现中，每个block处理一个cluster，多warp合并逻辑待完善
 */
void cuda_cos_topk_warpsort_fine_v1(
    const float* d_query_group,
    const float* d_cluster_vector,
    const int* d_cluster_query_offset,
    const int* d_cluster_query_data,
    const int* d_cluster_vector_index,
    const int* d_cluster_vector_num,
    const float* d_query_norm,
    const float* d_cluster_vector_norm,
    int* d_topk_index,
    float* d_topk_dist,
    int n_query,
    int distinct_cluster_count,
    int n_dim,
    int n_topn,
    int n_total_vectors
) {
    // 检查k是否在有效范围内
    if (n_topn > kMaxCapacity) {
        printf("Error: k (%d) exceeds maximum capacity (%d)\n", n_topn, kMaxCapacity);
        return;
    }
    
    // 选择合适的Capacity（必须是2的幂，且 > k）
    int capacity = 32;
    while (capacity < n_topn) capacity <<= 1;
    capacity = std::min(capacity, kMaxCapacity);  // 限制最大capacity
    
    // 配置kernel launch（借鉴cuVS线程模型）
    // grid.y = n_query: 每个block处理一个query
    // grid.x = 1: 每个block处理该query的所有相关cluster（block内的warps分摊）
    dim3 grid(1, n_query);  // 2D grid: (1, n_query)
    dim3 block(256);  // 256线程（8个warp）
    
    // 计算共享内存大小：query缓存 + warp局部topk存储
    const int query_smem_elems = (n_dim <= 128) ? n_dim : 128;
    const int n_warps = 256 / 32;  // 8 warps
    const int smem_size = query_smem_elems * sizeof(float) +  // query缓存
                          n_warps * n_topn * sizeof(float) +   // warp局部距离
                          n_warps * n_topn * sizeof(int);      // warp局部索引
    
    // printf("[DEBUG] Launching kernel: grid=(%d,%d), block=%d, n_query=%d, distinct_clusters=%d, n_dim=%d, k=%d, smem_size=%d\n", 
    //        1, n_query, 256, n_query, distinct_cluster_count, n_dim, n_topn, smem_size);
    // fflush(stdout);
    
    {
        CUDATimer timer_kernel("Indexed Inner Product with TopK Kernel (v1)", ENABLE_CUDA_TIMING);
        
        // 根据capacity选择kernel实例
        if (capacity <= 32) {
            indexed_inner_product_with_topk_kernel<64, true><<<grid, block, smem_size>>>(
                d_query_group,
                d_cluster_vector,
                d_cluster_query_offset,
                d_cluster_query_data,
                d_cluster_vector_index,
                d_cluster_vector_num,
                d_query_norm,
                d_cluster_vector_norm,
                n_query,
                distinct_cluster_count,
                n_dim,
                n_total_vectors,
                n_topn,
                d_topk_dist,
                d_topk_index
            );
        } else if (capacity <= 64) {
            indexed_inner_product_with_topk_kernel<128, true><<<grid, block, smem_size>>>(
                d_query_group,
                d_cluster_vector,
                d_cluster_query_offset,
                d_cluster_query_data,
                d_cluster_vector_index,
                d_cluster_vector_num,
                d_query_norm,
                d_cluster_vector_norm,
                n_query,
                distinct_cluster_count,
                n_dim,
                n_total_vectors,
                n_topn,
                d_topk_dist,
                d_topk_index
            );
        } else {
            indexed_inner_product_with_topk_kernel<256, true><<<grid, block, smem_size>>>(
                d_query_group,
                d_cluster_vector,
                d_cluster_query_offset,
                d_cluster_query_data,
                d_cluster_vector_index,
                d_cluster_vector_num,
                d_query_norm,
                d_cluster_vector_norm,
                n_query,
                distinct_cluster_count,
                n_dim,
                n_total_vectors,
                n_topn,
                d_topk_dist,
                d_topk_index
            );
        }
        
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[ERROR] Kernel execution failed: %s\n", cudaGetErrorString(err));
        }
        CHECK_CUDA_ERRORS;
        // printf("[DEBUG] Kernel execution completed\n");
        // fflush(stdout);
    }
}
