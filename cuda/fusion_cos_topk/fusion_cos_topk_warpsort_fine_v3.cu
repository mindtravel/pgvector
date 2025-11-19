#include "fusion_cos_topk.cuh"
#include "../indexed_gemm/indexed_gemm.cuh"
#include "../pch.h"
#include "../unit_tests/common/test_utils.cuh"
#include "../warpsortfilter/warpsort_utils.cuh"
#include <algorithm>

#define ENABLE_CUDA_TIMING 0
using namespace pgvector::warpsort_utils;

/**
 * 流式融合余弦距离top-k计算（v3版本：一个block处理多个query）
 * 
 * 相比v2版本的改进：
 * - 每个block处理多个query（默认8个），提高GPU利用率
 * - 减少kernel启动开销
 * - 每个warp独立处理一个query
 */
void cuda_cos_topk_warpsort_fine_v3(
    float* d_query_group,
    float* d_cluster_vector,
    int* d_query_index,
    
    float* d_query_norm,
    float* d_cluster_vector_norm,
    
    int* d_topk_index,
    float* d_topk_dist,
    
    int n_selected_querys,
    int n_selected_vectors,
    int n_dim,
    int k
) {
    // 检查k是否在有效范围内
    if (k > kMaxCapacity) {
        printf("Error: k (%d) exceeds maximum capacity (%d)\n", k, kMaxCapacity);
        return;
    }
    
    // 选择合适的Capacity（必须是2的幂，且 > k）
    int capacity = 32;
    while (capacity < k) capacity <<= 1;
    capacity = std::min(capacity, kMaxCapacity);
    
    // 配置kernel launch
    // v3版本：每个block处理8个query，每个warp处理一个query
    // block需要至少8*32=256个线程
    constexpr int kQueriesPerBlock = 8;
    dim3 block(256);  // 8个warp，每个warp 32个线程
    
    {
        CUDATimer timer_kernel("Indexed Inner Product with TopK Kernel (v3)", ENABLE_CUDA_TIMING);
        
        // 根据capacity选择kernel实例
        if (capacity <= 32) {
            launch_indexed_inner_product_with_topk_kernel_v3<64, true, kQueriesPerBlock>(
                block,
                n_dim,
                d_query_group,
                d_cluster_vector,
                d_query_index,
                d_query_norm,
                d_cluster_vector_norm,
                n_selected_querys,
                n_selected_vectors,
                k,
                d_topk_dist,
                d_topk_index
            );
        } else if (capacity <= 64) {
            launch_indexed_inner_product_with_topk_kernel_v3<128, true, kQueriesPerBlock>(
                block,
                n_dim,
                d_query_group,
                d_cluster_vector,
                d_query_index,
                d_query_norm,
                d_cluster_vector_norm,
                n_selected_querys,
                n_selected_vectors,
                k,
                d_topk_dist,
                d_topk_index
            );
        } else {
            launch_indexed_inner_product_with_topk_kernel_v3<256, true, kQueriesPerBlock>(
                block,
                n_dim,
                d_query_group,
                d_cluster_vector,
                d_query_index,
                d_query_norm,
                d_cluster_vector_norm,
                n_selected_querys,
                n_selected_vectors,
                k,
                d_topk_dist,
                d_topk_index
            );
        }
        
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[ERROR] Kernel execution failed: %s\n", cudaGetErrorString(err));
        }
        CHECK_CUDA_ERRORS;
    }
}

