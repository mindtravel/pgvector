#include "fine_screen_top_n.cuh"
#include "../l2norm/l2norm.cuh"
#include "../unit_tests/common/test_utils.cuh"
#include "../fusion_cos_topk/fusion_cos_topk.cuh"
#include "../warpsortfilter/warpsort_topk.cu"
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

using namespace pgvector::warpsort_topk;

/**
 * 简单的双流流水线（支持 cuda_cos_topk_warpsort_fine_v3_fixed_probe）
 * 流1：数据上传
 * 流2：计算处理
 * 
 * 参考 fine_screen_top_n.cu 的实现，使用相同的数据结构和kernel调用
 */
void simple_dual_stream_pipeline(
    void* cluster_data_array_ptr, int num_batches, int batch_size,
    float* h_query_group, int n_query, int n_dim, int k,
    int* h_query_topn_index, float* h_query_topn_dist) {
    
    // 转换指针类型（保持兼容性）
    ClusterQueryData* cluster_data_array = (ClusterQueryData*)cluster_data_array_ptr;
    
    // 创建两个CUDA流
    cudaStream_t upload_stream, compute_stream;
    cudaStreamCreate(&upload_stream);
    cudaStreamCreate(&compute_stream);
    
    // 创建同步事件
    cudaEvent_t upload_done;
    cudaEventCreate(&upload_done);
    
    printf("Processing %d batches with batch_size=%d using dual streams\n", num_batches, batch_size);
    
    // 处理每个batch
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        printf("Processing batch %d/%d\n", batch_idx + 1, num_batches);
        
        ClusterQueryData& current_batch = cluster_data_array[batch_idx];
        
        // 从 ClusterQueryData 转换为 fine_screen_top_n_blocks 需要的格式
        // 计算当前batch的probe数量（假设每个cluster是一个probe）
        int n_probes = batch_size;
        
        // 准备 block 数据
        float** h_block_vectors = (float**)malloc(n_probes * sizeof(float*));
        int* h_block_vector_counts = (int*)malloc(n_probes * sizeof(int));
        int* h_block_query_offset = (int*)malloc((n_probes + 1) * sizeof(int));
        std::vector<int> h_block_query_data_vec;
        std::vector<int> h_block_query_probe_indices_vec;
        
        int query_offset = 0;
        h_block_query_offset[0] = 0;
        
        for (int i = 0; i < n_probes; i++) {
            // 获取cluster的向量数据
            int cluster_id = current_batch.cluster_map[i];
            int vector_count = current_batch.cluster_vector_num[i];
            h_block_vector_counts[i] = vector_count;
            
            // 分配并复制向量数据
            h_block_vectors[i] = (float*)malloc(vector_count * n_dim * sizeof(float));
            for (int j = 0; j < vector_count; j++) {
                memcpy(h_block_vectors[i] + j * n_dim, 
                       current_batch.cluster_vector[i][j], 
                       n_dim * sizeof(float));
            }
            
            // 获取query列表
            int query_start = current_batch.cluster_query_offset[i];
            int query_end = current_batch.cluster_query_offset[i + 1];
            int n_queries = query_end - query_start;
            
            for (int j = 0; j < n_queries; j++) {
                int query_id = current_batch.cluster_query_data[query_start + j];
                h_block_query_data_vec.push_back(query_id);
                h_block_query_probe_indices_vec.push_back(i);  // probe在query中的索引
            }
            
            query_offset += n_queries;
            h_block_query_offset[i + 1] = query_offset;
        }
        
        int* h_block_query_data = h_block_query_data_vec.data();
        int* h_block_query_probe_indices = h_block_query_probe_indices_vec.data();
        
        // 计算总向量数
        int total_vectors = 0;
        for (int i = 0; i < n_probes; i++) {
            total_vectors += h_block_vector_counts[i];
        }
        
        // 分配GPU内存（流1：异步上传）
        float* d_cluster_vector = nullptr;
        float* d_query_group = nullptr;
        int* d_probe_vector_offset = nullptr;
        int* d_probe_vector_count = nullptr;
        int* d_probe_queries = nullptr;
        int* d_probe_query_offsets = nullptr;
        int* d_probe_query_probe_indices = nullptr;
        float* d_query_norm = nullptr;
        float* d_cluster_vector_norm = nullptr;
        float* d_topk_dist_final = nullptr;
        int* d_topk_index_final = nullptr;
        
        // 计算block vector offset
        int* block_vector_offset = (int*)malloc((n_probes + 1) * sizeof(int));
        int vec_offset = 0;
        for (int i = 0; i < n_probes; i++) {
            block_vector_offset[i] = vec_offset;
            vec_offset += h_block_vector_counts[i];
        }
        block_vector_offset[n_probes] = vec_offset;
        
        // 流1：异步上传数据
        cudaMalloc(&d_cluster_vector, total_vectors * n_dim * sizeof(float));
        float* d_cluster_vector_ptr = d_cluster_vector;
        for (int i = 0; i < n_probes; i++) {
            int vec_count = h_block_vector_counts[i];
            if (vec_count > 0 && h_block_vectors[i]) {
                cudaMemcpyAsync(d_cluster_vector_ptr, h_block_vectors[i], 
                               vec_count * n_dim * sizeof(float), 
                               cudaMemcpyHostToDevice, upload_stream);
                d_cluster_vector_ptr += vec_count * n_dim;
            }
        }
        
        cudaMalloc(&d_query_group, n_query * n_dim * sizeof(float));
        cudaMemcpyAsync(d_query_group, h_query_group, 
                       n_query * n_dim * sizeof(float), 
                       cudaMemcpyHostToDevice, upload_stream);
        
        cudaMalloc(&d_probe_vector_offset, n_probes * sizeof(int));
        cudaMemcpyAsync(d_probe_vector_offset, block_vector_offset, 
                       n_probes * sizeof(int), 
                       cudaMemcpyHostToDevice, upload_stream);
        
        cudaMalloc(&d_probe_vector_count, n_probes * sizeof(int));
        cudaMemcpyAsync(d_probe_vector_count, h_block_vector_counts, 
                       n_probes * sizeof(int), 
                       cudaMemcpyHostToDevice, upload_stream);
        
        int total_queries_in_blocks = h_block_query_offset[n_probes];
        cudaMalloc(&d_probe_queries, total_queries_in_blocks * sizeof(int));
        cudaMemcpyAsync(d_probe_queries, h_block_query_data, 
                       total_queries_in_blocks * sizeof(int), 
                       cudaMemcpyHostToDevice, upload_stream);
        
        cudaMalloc(&d_probe_query_offsets, (n_probes + 1) * sizeof(int));
        cudaMemcpyAsync(d_probe_query_offsets, h_block_query_offset, 
                       (n_probes + 1) * sizeof(int), 
                       cudaMemcpyHostToDevice, upload_stream);
        
        cudaMalloc(&d_probe_query_probe_indices, total_queries_in_blocks * sizeof(int));
        cudaMemcpyAsync(d_probe_query_probe_indices, h_block_query_probe_indices, 
                       total_queries_in_blocks * sizeof(int), 
                       cudaMemcpyHostToDevice, upload_stream);
        
        // 分配norm内存
        cudaMalloc(&d_query_norm, n_query * sizeof(float));
        cudaMalloc(&d_cluster_vector_norm, total_vectors * sizeof(float));
        
        // 分配输出内存
        cudaMalloc(&d_topk_dist_final, n_query * k * sizeof(float));
        cudaMalloc(&d_topk_index_final, n_query * k * sizeof(int));
        
        // 记录上传完成事件
        cudaEventRecord(upload_done, upload_stream);
        
        // 流2：等待上传完成后进行计算
        cudaStreamWaitEvent(compute_stream, upload_done, 0);
        
        // 计算L2 norm（在计算流中）
        compute_l2_norm_gpu(d_query_group, d_query_norm, n_query, n_dim);
        compute_l2_norm_gpu(d_cluster_vector, d_cluster_vector_norm, total_vectors, n_dim);
        
        // 调用 cuda_cos_topk_warpsort_fine_v3_fixed_probe（在计算流中）
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
            d_topk_index_final,
            d_topk_dist_final,
            nullptr,  // candidate_dist
            nullptr,  // candidate_index
            n_query,
            n_probes,  // n_total_clusters (每个probe对应一个cluster)
            n_probes,
            n_dim,
            k
        );
        
        // 下载结果（在计算流中）
        cudaMemcpyAsync(h_query_topn_index, d_topk_index_final, 
                       n_query * k * sizeof(int), 
                       cudaMemcpyDeviceToHost, compute_stream);
        
        cudaMemcpyAsync(h_query_topn_dist, d_topk_dist_final, 
                       n_query * k * sizeof(float), 
                       cudaMemcpyDeviceToHost, compute_stream);
        
        // 等待计算完成
        cudaStreamSynchronize(compute_stream);
        
        // 清理GPU内存
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
        
        // 清理主机内存
        for (int i = 0; i < n_probes; i++) {
            if (h_block_vectors[i]) {
                free(h_block_vectors[i]);
            }
        }
        free(h_block_vectors);
        free(h_block_vector_counts);
        free(h_block_query_offset);
        free(block_vector_offset);
        
        printf("Batch %d completed\n", batch_idx + 1);
    }
    
    // 清理流和事件
    cudaStreamDestroy(upload_stream);
    cudaStreamDestroy(compute_stream);
    cudaEventDestroy(upload_done);
    
    printf("All batches processed successfully!\n");
}
