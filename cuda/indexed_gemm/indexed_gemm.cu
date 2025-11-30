#include "indexed_gemm.cuh"
#include "../pch.h"
// ============================================================================
// Indexed Inner Product Kernel for Fine Screening
// ============================================================================


/**
 * 索引化内积计算kernel（使用连续存储）
 * 
 * 每个block处理一个cluster，计算该cluster对应的query与cluster向量的内积。
 * 向量按聚类物理连续存储，使用连续索引访问（符合pgvector实现）。
 * 
 * 使用max_candidates_per_query作为内存布局维度（而非max_cluster_vector_count），
 * 这样可以使用实际的最大候选数，减少内存分配。
 */
__global__ void indexed_inner_product_kernel(
    const float* __restrict__ d_query_group,
    const float* __restrict__ d_cluster_vector,
    const int* __restrict__ d_cluster_query_offset,
    const int* __restrict__ d_cluster_query_data,
    const int* __restrict__ d_cluster_vector_index,  // 每个cluster在全局向量数组中的连续起始位置
    const int* __restrict__ d_cluster_vector_num,
    float* __restrict__ d_inner_product,
    int* __restrict__ d_index,
    int* __restrict__ d_query_count,  // 每个query当前的候选数量（使用原子操作）
    const int* __restrict__ d_num_samples,  // 每个query的实际候选数量 [n_query]（用于边界检查）
    int n_query,
    int distinct_cluster_count,
    int n_dim,
    int tol_vector,
    int max_candidates_per_query  // 每个query的最大候选数（基于实际计算，用于内存布局）
) {
    const int cluster_idx = blockIdx.x;
    if (cluster_idx >= distinct_cluster_count) return;
    
    const int thread_idx = threadIdx.x;
    const int block_dim = blockDim.x;
    
    // 获取当前cluster的query范围
    // 使用标准的offset数组格式：offset数组有 distinct_cluster_count + 1 个元素
    // count = offset[i+1] - offset[i]
    int query_start = d_cluster_query_offset[cluster_idx];
    int query_end = d_cluster_query_offset[cluster_idx + 1];
    int query_count = query_end - query_start;
    
    if (query_count <= 0) return;
    
    // 获取当前cluster的向量信息（连续存储）
    int vector_start_idx = d_cluster_vector_index[cluster_idx];  // cluster的连续起始位置
    int vector_count = d_cluster_vector_num[cluster_idx];
    
    // 边界检查
    if (vector_start_idx < 0 || vector_count <= 0 || 
        vector_start_idx + vector_count > tol_vector) {
        return;
    }
    
    // 每个线程处理部分query和cluster向量的内积计算
    // 外层循环：遍历该cluster对应的query
    for (int q = 0; q < query_count; q++) {
        int query_idx = d_cluster_query_data[query_start + q];
        
        // 边界检查
        if (query_idx < 0 || query_idx >= n_query) continue;
        
        // 内层循环：遍历cluster中的向量（连续索引）
        for (int vec_idx = thread_idx; vec_idx < vector_count; vec_idx += block_dim) {
            int global_vec_idx = vector_start_idx + vec_idx;  // 连续索引
            
            // 边界检查
            if (global_vec_idx < 0 || global_vec_idx >= tol_vector) continue;
            
            // 计算内积
            float dot_product = 0.0f;
            #pragma unroll 4
            for (int dim = 0; dim < n_dim; dim++) {
                dot_product += d_query_group[query_idx * n_dim + dim] * 
                              d_cluster_vector[global_vec_idx * n_dim + dim];
            }
            
            // 使用原子操作获取该query的下一个存储位置
            int pos = atomicAdd(&d_query_count[query_idx], 1);
            
            // 边界检查：使用该query的实际候选数（而非全局最大值）
            int actual_num_samples = d_num_samples[query_idx];
            if (pos < actual_num_samples && pos < max_candidates_per_query) {
                int output_idx = query_idx * max_candidates_per_query + pos;
                d_inner_product[output_idx] = dot_product;
                d_index[output_idx] = global_vec_idx;  // 存储全局向量索引
            }
        }
    }
}