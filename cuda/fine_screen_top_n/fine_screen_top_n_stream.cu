#include "fine_screen_top_n.cuh"
#include "../l2norm.cuh"
#include "../unit_tests/common/test_utils.cuh"
#include <cuda_runtime.h>

/**
 * 简单的双流流水线
 * 流1：数据上传
 * 流2：计算处理
 */
void simple_dual_stream_pipeline(
    ClusterQueryData* cluster_data_array, int num_batches, int batch_size,
    float* h_query_group, int n_query, int n_dim, int n_topn,
    int* h_query_topn_index, float* h_query_topn_dist) {
    
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
        
        // 计算当前batch的数据量
        int total_queries = 0;
        int total_vectors = 0;
        for (int i = 0; i < batch_size; i++) {
            total_queries += current_batch.cluster_query_data_size[0];
            total_vectors += current_batch.cluster_vector_num[0];
        }
        
        // 分配GPU内存
        float* d_cluster_vector;
        int* d_cluster_query_data;
        int* d_cluster_query_offset;
        int* d_cluster_map;
        int* d_cluster_vector_index;
        int* d_cluster_vector_num;
        float* d_query_group;
        int* d_query_topn_index;
        float* d_query_topn_dist;
        
        cudaMalloc(&d_cluster_vector, total_vectors * n_dim * sizeof(float));
        cudaMalloc(&d_cluster_query_data, total_queries * sizeof(int));
        cudaMalloc(&d_cluster_query_offset, batch_size * sizeof(int));
        cudaMalloc(&d_cluster_map, batch_size * sizeof(int));
        cudaMalloc(&d_cluster_vector_index, batch_size * sizeof(int));
        cudaMalloc(&d_cluster_vector_num, batch_size * sizeof(int));
        cudaMalloc(&d_query_group, n_query * n_dim * sizeof(float));
        cudaMalloc(&d_query_topn_index, n_query * n_topn * sizeof(int));
        cudaMalloc(&d_query_topn_dist, n_query * n_topn * sizeof(float));
        
        // 流1：异步上传数据
        cudaMemcpyAsync(d_cluster_map, current_batch.cluster_map, 
                       batch_size * sizeof(int), cudaMemcpyHostToDevice, upload_stream);
        
        cudaMemcpyAsync(d_cluster_query_offset, current_batch.cluster_query_offset, 
                       batch_size * sizeof(int), cudaMemcpyHostToDevice, upload_stream);
        
        cudaMemcpyAsync(d_cluster_query_data, current_batch.cluster_query_data, 
                       total_queries * sizeof(int), cudaMemcpyHostToDevice, upload_stream);
        
        cudaMemcpyAsync(d_cluster_vector_index, current_batch.cluster_vector_index, 
                       batch_size * sizeof(int), cudaMemcpyHostToDevice, upload_stream);
        
        cudaMemcpyAsync(d_cluster_vector_num, current_batch.cluster_vector_num, 
                       batch_size * sizeof(int), cudaMemcpyHostToDevice, upload_stream);
        
        // 上传向量数据
        size_t vector_offset = 0;
        for (int i = 0; i < batch_size; i++) {
            int vector_count = current_batch.cluster_vector_num[0];
            for (int j = 0; j < vector_count; j++) {
                cudaMemcpyAsync(d_cluster_vector + vector_offset, 
                               current_batch.cluster_vector[0][j], 
                               n_dim * sizeof(float), 
                               cudaMemcpyHostToDevice, upload_stream);
                vector_offset += n_dim;
            }
        }
        
        // 上传query数据
        cudaMemcpyAsync(d_query_group, h_query_group, 
                       n_query * n_dim * sizeof(float), 
                       cudaMemcpyHostToDevice, upload_stream);
        
        // 记录上传完成事件
        cudaEventRecord(upload_done, upload_stream);
        
        // 流2：等待上传完成后进行计算
        cudaStreamWaitEvent(compute_stream, upload_done, 0);
        
        // 初始化输出数据
        cudaMemsetAsync(d_query_topn_index, -1, n_query * n_topn * sizeof(int), compute_stream);
        cudaMemsetAsync(d_query_topn_dist, FLT_MAX, n_query * n_topn * sizeof(float), compute_stream);
        
        // TODO: 在这里调用实际的计算kernel
        // 目前先空出来，等待具体计算实现
        
        // 下载结果
        cudaMemcpyAsync(h_query_topn_index, d_query_topn_index, 
                       n_query * n_topn * sizeof(int), 
                       cudaMemcpyDeviceToHost, compute_stream);
        
        cudaMemcpyAsync(h_query_topn_dist, d_query_topn_dist, 
                       n_query * n_topn * sizeof(float), 
                       cudaMemcpyDeviceToHost, compute_stream);
        
        // 等待计算完成
        cudaStreamSynchronize(compute_stream);
        
        // 清理GPU内存
        cudaFree(d_cluster_vector);
        cudaFree(d_cluster_query_data);
        cudaFree(d_cluster_query_offset);
        cudaFree(d_cluster_map);
        cudaFree(d_cluster_vector_index);
        cudaFree(d_cluster_vector_num);
        cudaFree(d_query_group);
        cudaFree(d_query_topn_index);
        cudaFree(d_query_topn_dist);
        
        printf("Batch %d completed\n", batch_idx + 1);
    }
    
    // 清理流和事件
    cudaStreamDestroy(upload_stream);
    cudaStreamDestroy(compute_stream);
    cudaEventDestroy(upload_done);
    
    printf("All batches processed successfully!\n");
}
