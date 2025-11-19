#include "fusion_cos_topk.cuh"
#include "../indexed_gemm/indexed_gemm.cuh"
#include "../pch.h"
#include "../unit_tests/common/test_utils.cuh"
#include "../warpsortfilter/warpsort_utils.cuh"
#include <algorithm>

#define ENABLE_CUDA_TIMING 0
using namespace pgvector::warpsort_utils;

/**
 * 流式融合余弦距离top-k计算（v2版本：优化数据上传）
 * 
 * 新设计特点：
 * 1. 只上传涉及的cluster向量（而非所有cluster）
 * 2. 使用query到cluster的映射（CSR格式）
 * 3. 预计算query和cluster向量的l2norm
 * 4. cluster向量在GPU上连续存储（只包含涉及的cluster）
 * 
 * 关键改进：
 * - 减少显存占用：只上传需要的cluster向量
 * - 简化数据组织：使用query_cluster_offset和cluster_vector_offset进行索引
 * - 预计算l2norm：避免在kernel中重复计算
 * 
 * 限制：
 * - k <= 256 (warp-sort容量限制)
 * 
 * @param d_query_group query向量 [n_query * n_dim]
 * @param d_cluster_vector 涉及的cluster向量（连续存储）[n_selected_vectors * n_dim]
 * @param d_query_cluster_offset query到cluster映射的offset [n_query+1]
 * @param d_query_cluster_data query到cluster映射的data [total_relations]
 * @param d_cluster_vector_offset 每个cluster在d_cluster_vector中的起始位置 [n_selected_clusters+1]
 * @param d_query_norm query的l2norm [n_query]
 * @param d_cluster_vector_norm cluster向量的l2norm [n_selected_vectors]
 * @param d_topk_index [out] 每个query的topk索引 [n_query * k]
 * @param d_topk_dist [out] 每个query的topk距离 [n_query * k]
 * @param n_query query数量
 * @param n_selected_clusters 涉及的cluster数量
 * @param n_selected_vectors 涉及的向量总数
 * @param n_dim 向量维度
 * @param k topk数量
 */
void cuda_cos_topk_warpsort_fine_v2(
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
    // grid.y = n_query: 每个block处理一个query
    // grid.x = 1: 每个block处理该query的所有相关cluster（block内的warps分摊）
    dim3 grid(1, n_selected_querys);
    dim3 block(32);  // 256线程（8个warp）
    
    {
        CUDATimer timer_kernel("Indexed Inner Product with TopK Kernel (v2)", ENABLE_CUDA_TIMING);
        
        // 根据capacity选择kernel实例
        if (capacity <= 32) {
            // indexed_inner_product_with_topk_kernel_v2<64, true><<<grid, block, smem_size>>> (
            launch_indexed_inner_product_with_topk_kernel_v2<64, true>(
                grid, block,

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
            // indexed_inner_product_with_topk_kernel_v2<128, true><<<grid, block, smem_size>>> (
            launch_indexed_inner_product_with_topk_kernel_v2<128, true>(
                grid, block,

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
            // indexed_inner_product_with_topk_kernel_v2<256, true><<<grid, block, smem_size>>> (
            launch_indexed_inner_product_with_topk_kernel_v2<256, true>(
                grid, block,

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

