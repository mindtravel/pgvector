#include "fine_screen_top_n.cuh"
#include "../l2norm/l2norm.cuh"
#include "../unit_tests/common/test_utils.cuh"
#include "../fusion_cos_topk/fusion_cos_topk.cuh"
#include "../warpsortfilter/warpsort_topk.cu"
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <limits.h>
#include <float.h>
#include <algorithm>
#include <vector>

using namespace pgvector::warpsort_topk;

#define ENABLE_CUDA_TIMING 1

/**
 * 精筛 top-n（固定 probe 版本）
 * 
 * 使用固定 probe 方案，每个 block 处理一个 probe 的多个 query
 * 
 * @param h_query_group query向量 [n_query * n_dim]
 * @param h_block_vectors block向量数据指针数组 [n_probes]
 * @param h_block_vector_counts 每个block的向量数量 [n_probes]
 * @param h_block_query_offset block对应的query列表offset（CSR格式）[n_probes + 1]
 * @param h_block_query_data block对应的query列表（CSR格式）[total_queries]
 * @param h_query_topk_index [out] 每个query的topk索引 [n_query * k]
 * @param h_query_topk_dist [out] 每个query的topk距离 [n_query * k]
 * @param n_query query数量
 * @param n_probes probe数量
 * @param n_dim 向量维度
 * @param k top-n数量
 */
void fine_screen_topk(
    float* h_query_group,

    float** h_block_vectors,
    int* h_block_vector_counts,
    int* h_block_query_offset,  // 大小为(n_total_clusters + 1)，包含所有cluster
    int* h_block_query_data,
    int* h_block_query_probe_indices,  // 新增：每个block-query对中probe在query中的索引
    int** h_query_topk_index,
    float** h_query_topk_dist,

    int n_query,
    int n_total_clusters,  // 总的cluster数量
    int n_probes,  // 每个query的probe数量
    int n_dim,
    int k
) {
    if (n_total_clusters <= 0 || n_query <= 0 || n_dim <= 0 || k <= 0) {
        return;
    }
    
    // 1. 构建probe_vector_offset和probe_vector_count（完整的数组，包含所有cluster）
    // 计算总向量数和每个cluster的向量offset
    int* probe_vector_offset = (int*)malloc(n_total_clusters * sizeof(int));
    int* probe_vector_count = (int*)malloc(n_total_clusters * sizeof(int));
    int total_vectors = 0;
    
    for (int i = 0; i < n_total_clusters; ++i) {
        probe_vector_offset[i] = total_vectors;
        probe_vector_count[i] = h_block_vector_counts[i];  // 直接使用，未使用的cluster为0
        total_vectors += h_block_vector_counts[i];
    }
    
    // 2. 将所有cluster的向量数据上传到GPU（连续存储）
    float* d_cluster_vector = nullptr;
    cudaMalloc(&d_cluster_vector, total_vectors * n_dim * sizeof(float));
    CHECK_CUDA_ERRORS;
    
    float* d_cluster_vector_ptr = d_cluster_vector;
    for (int i = 0; i < n_total_clusters; ++i) {
        int vec_count = h_block_vector_counts[i];
        if (vec_count > 0 && h_block_vectors[i]) {
            cudaMemcpy(d_cluster_vector_ptr, h_block_vectors[i], vec_count * n_dim * sizeof(float), cudaMemcpyHostToDevice);
            d_cluster_vector_ptr += vec_count * n_dim;
        }
    }
    CHECK_CUDA_ERRORS;
    
    // 3. 上传query数据
    float* d_query_group = nullptr;
    cudaMalloc(&d_query_group, n_query * n_dim * sizeof(float));
    cudaMemcpy(d_query_group, h_query_group, n_query * n_dim * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;
    
    // 4. 上传probe（cluster）信息到GPU
    int* d_probe_vector_offset = nullptr;
    int* d_probe_vector_count = nullptr;
    int* d_probe_queries = nullptr;
    int* d_probe_query_offsets = nullptr;
    
    cudaMalloc(&d_probe_vector_offset, n_total_clusters * sizeof(int));
    cudaMalloc(&d_probe_vector_count, n_total_clusters * sizeof(int));
    cudaMalloc(&d_probe_query_offsets, (n_total_clusters + 1) * sizeof(int));
    
    int total_queries_in_blocks = h_block_query_offset[n_total_clusters];
    cudaMalloc(&d_probe_queries, total_queries_in_blocks * sizeof(int));
    
    cudaMemcpy(d_probe_vector_offset, probe_vector_offset, 
               n_total_clusters * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_vector_count, probe_vector_count, 
               n_total_clusters * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_query_offsets, h_block_query_offset, 
               (n_total_clusters + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;
    
    cudaMemcpy(d_probe_queries, h_block_query_data, 
               total_queries_in_blocks * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;
    
    // 5. 上传probe在query中的索引到GPU（已在Step 2中构建，直接使用）
    int* d_probe_query_probe_indices = nullptr;
    cudaMalloc(&d_probe_query_probe_indices, total_queries_in_blocks * sizeof(int));
    cudaMemcpy(d_probe_query_probe_indices, h_block_query_probe_indices,
               total_queries_in_blocks * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;
    
    // 计算L2 norm
    float* d_query_norm = nullptr;
    float* d_cluster_vector_norm = nullptr;
    
    cudaMalloc(&d_query_norm, n_query * sizeof(float));
    cudaMalloc(&d_cluster_vector_norm, total_vectors * sizeof(float));
    
    // 计算query norm和cluster vector norm
    compute_l2_norm_gpu(d_query_group, d_query_norm, n_query, n_dim);
    compute_l2_norm_gpu(d_cluster_vector, d_cluster_vector_norm, total_vectors, n_dim);
    CHECK_CUDA_ERRORS;
    
    // 6. 调用固定probe版本的精筛kernel（包含规约，输出直接是 [n_query][k]）
    // 分配最终输出缓冲区
    float* d_topk_dist_final = nullptr;
    int* d_topk_index_final = nullptr;
    cudaMalloc(&d_topk_dist_final, n_query * k * sizeof(float));
    cudaMalloc(&d_topk_index_final, n_query * k * sizeof(int));

    cuda_cos_topk_warpsort_fine(
        d_query_group,
        d_cluster_vector,
        
        d_probe_vector_offset,
        d_probe_vector_count,
        d_probe_queries,
        d_probe_query_offsets,
        d_probe_query_probe_indices,
        
        d_query_norm,
        d_cluster_vector_norm,
        
        d_topk_index_final,  // 最终输出：[n_query][k]
        d_topk_dist_final,   // 最终输出：[n_query][k]
        
        nullptr,
        nullptr,

        n_query,
        n_total_clusters,  // 总的cluster数量
        n_probes,  // 每个query的probe数量
        n_dim,
        k
    );
    CHECK_CUDA_ERRORS;

    // 7. 从GPU读取最终结果（规约已在函数内部完成）
    cudaMemcpy(h_query_topk_dist[0], d_topk_dist_final, 
               n_query * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_query_topk_index[0], d_topk_index_final, 
               n_query * k * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;
    
    // 8. 清理GPU内存
    cudaFree(d_cluster_vector);
    cudaFree(d_query_group);
    cudaFree(d_probe_vector_offset);
    cudaFree(d_probe_vector_count);
    cudaFree(d_probe_queries);
    cudaFree(d_probe_query_offsets);
    cudaFree(d_probe_query_probe_indices);
    cudaFree(d_query_norm);
    cudaFree(d_cluster_vector_norm);
    cudaFree(d_topk_dist_final);
    cudaFree(d_topk_index_final);
    free(probe_vector_offset);
    free(probe_vector_count);
    CHECK_CUDA_ERRORS;
}