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
 * @param h_query_topn_index [out] 每个query的topn索引 [n_query * k]
 * @param h_query_topn_dist [out] 每个query的topn距离 [n_query * k]
 * @param n_query query数量
 * @param n_probes probe数量
 * @param n_dim 向量维度
 * @param k top-n数量
 */
void fine_screen_top_n_blocks(
    float* h_query_group,

    float** h_block_vectors,
    int* h_block_vector_counts,
    int* h_block_query_offset,
    int* h_block_query_data,
    int* h_block_query_probe_indices,  // 新增：每个block-query对中probe在query中的索引
    int* h_query_topn_index,
    float* h_query_topn_dist,

    float** candidate_dist,
    int** candidate_index,

    int n_query,
    int n_probes,
    int n_dim,
    int k
) {
    if (n_probes <= 0 || n_query <= 0 || n_dim <= 0 || k <= 0) {
        return;
    }
    
    // 1. 计算总向量数和每个block的向量offset
    int* block_vector_offset = (int*)malloc((n_probes + 1) * sizeof(int));
    int total_vectors = 0;
    int max_queries_per_block = 0;
    for (int i = 0; i < n_probes; ++i) {
        block_vector_offset[i] = total_vectors;
        total_vectors += h_block_vector_counts[i];
        int n_queries = h_block_query_offset[i + 1] - h_block_query_offset[i];
        max_queries_per_block = std::max(max_queries_per_block, n_queries);
    }
    block_vector_offset[n_probes] = total_vectors;
    
    // 2. 将所有block的向量数据上传到GPU（连续存储）
    float* d_cluster_vector = nullptr;
    cudaMalloc(&d_cluster_vector, total_vectors * n_dim * sizeof(float));
    CHECK_CUDA_ERRORS;
    
    float* d_cluster_vector_ptr = d_cluster_vector;
    for (int i = 0; i < n_probes; ++i) {
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
    
    // 4. 上传probe（block）信息到GPU
    int* d_probe_vector_offset = nullptr;
    int* d_probe_vector_count = nullptr;
    int* d_probe_queries = nullptr;
    int* d_probe_query_offsets = nullptr;
    
    cudaError_t err1 = cudaMalloc(&d_probe_vector_offset, n_probes * sizeof(int));
    cudaError_t err2 = cudaMalloc(&d_probe_vector_count, n_probes * sizeof(int));
    cudaError_t err3 = cudaMalloc(&d_probe_query_offsets, (n_probes + 1) * sizeof(int));
    
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        printf("[ERROR] Failed to allocate probe metadata: err1=%s, err2=%s, err3=%s\n",
               cudaGetErrorString(err1), cudaGetErrorString(err2), cudaGetErrorString(err3));
        return;
    }
    
    printf("[DEBUG] Allocated d_probe_query_offsets=%p, size=%zu bytes\n", 
           d_probe_query_offsets, (n_probes + 1) * sizeof(int));
    
    cudaMemcpy(d_probe_vector_offset, block_vector_offset, 
               n_probes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_vector_count, h_block_vector_counts, 
               n_probes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_query_offsets, h_block_query_offset, 
               (n_probes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    int total_queries_in_blocks = h_block_query_offset[n_probes];
    cudaError_t err4 = cudaMalloc(&d_probe_queries, total_queries_in_blocks * sizeof(int));
    if (err4 != cudaSuccess) {
        printf("[ERROR] Failed to allocate d_probe_queries: %s\n", cudaGetErrorString(err4));
        cudaFree(d_probe_vector_offset);
        cudaFree(d_probe_vector_count);
        cudaFree(d_probe_query_offsets);
        return;
    }
    
    printf("[DEBUG] Allocated d_probe_queries=%p, size=%zu bytes\n", 
           d_probe_queries, total_queries_in_blocks * sizeof(int));
    
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
    cudaError_t err_dist = cudaMalloc(&d_topk_dist_final, n_query * k * sizeof(float));
    cudaError_t err_index = cudaMalloc(&d_topk_index_final, n_query * k * sizeof(int));
    if (err_dist != cudaSuccess || err_index != cudaSuccess) {
        printf("[ERROR] Failed to allocate final output buffers: dist_err=%s, index_err=%s\n",
               cudaGetErrorString(err_dist), cudaGetErrorString(err_index));
        // 清理已分配的内存
        if (d_topk_dist_final) cudaFree(d_topk_dist_final);
        if (d_topk_index_final) cudaFree(d_topk_index_final);
        // 清理其他已分配的内存
        cudaFree(d_cluster_vector);
        cudaFree(d_query_group);
        cudaFree(d_probe_vector_offset);
        cudaFree(d_probe_vector_count);
        cudaFree(d_probe_queries);
        cudaFree(d_probe_query_offsets);
        cudaFree(d_probe_query_probe_indices);
        cudaFree(d_query_norm);
        cudaFree(d_cluster_vector_norm);
        free(block_vector_offset);
        return;
    }

    printf("[DEBUG] Before calling cuda_cos_topk_warpsort_fine_v3_fixed_probe:\n");
    printf("[DEBUG]   d_probe_query_offsets=%p\n", d_probe_query_offsets);
    printf("[DEBUG]   d_probe_queries=%p\n", d_probe_queries);
    fflush(stdout);
    
    cuda_cos_topk_warpsort_fine_v3_fixed_probe(
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
        
        candidate_dist,
        candidate_index,

        n_query,
        n_probes,
        n_dim,
        k
    );
        CHECK_CUDA_ERRORS;

    // 7. 从GPU读取最终结果（规约已在函数内部完成，无需CPU-GPU数据复制）
    cudaMemcpy(h_query_topn_dist, d_topk_dist_final, 
               n_query * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_query_topn_index, d_topk_index_final, 
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
    free(block_vector_offset);
    CHECK_CUDA_ERRORS;
}