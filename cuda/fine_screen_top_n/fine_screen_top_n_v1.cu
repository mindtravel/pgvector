#include "fine_screen_top_n.cuh"
#include "../l2norm/l2norm.cuh"
#include "../unit_tests/common/test_utils.cuh"
#include "../fusion_cos_topk/fusion_cos_topk.cuh"
#include <limits.h>
#include <float.h>

#define ENABLE_CUDA_TIMING 1

/**
 * 精筛topn融合算子（v1版本：流式计算）
 * 
 * 使用流式计算方式，在kernel内部维护warp-sort queue，直接写入最终结果。
 * 内存占用从 O(n_query * max_candidates) 降至 O(n_query * k)。
 * 
 * 参数说明（与fine_screen_top_n相同）:
 * - h_query_group: query向量组（二维指针）
 * - h_cluster_query_offset: cluster-query倒排索引的offset数组（标准格式：n_total_clusters+1个元素）
 * - h_cluster_query_data: cluster-query倒排索引的数据（连续存储）
 * - h_cluster_vector_index: 每个cluster在全局向量数组中的连续起始位置 [n_total_clusters]
 * - h_cluster_vector_num: 每个cluster的向量数量 [n_total_clusters]
 * - h_cluster_vector: 所有向量数据，按聚类物理连续存储 [n_total_vectors][n_dim]
 * - n_query: query总数
 * - n_cluster: 每个query精筛的cluster数量
 * - n_total_clusters: 所有distinct cluster数（用于offset数组大小）
 * - n_dim: 向量维度
 * - n_topn: 筛选Top-N（K）个（必须 <= 256）
 * - n_total_vectors: 所有向量总数
 * 
 * 出参:
 * - h_query_topn_index: query对应的topn向量的原始索引（二维指针）
 * - h_query_topn_dist: query对应的topn向量的距离（二维指针）
 * 
 * 限制：
 * - n_topn <= 256 (warp-sort容量限制)
 */
void fine_screen_top_n_v1(
    float** h_query_group, int* h_cluster_query_offset, int* h_cluster_query_data,
    int* h_cluster_vector_index, int* h_cluster_vector_num, float** h_cluster_vector,
    int n_query, int n_cluster, int n_total_clusters, int n_dim, int n_topn, int n_total_vectors,
    int** h_query_topn_index, float** h_query_topn_dist
) {
    // 检查k是否在有效范围内
    if (n_topn > 256) {
        printf("Error: fine_screen_top_n_v1 requires n_topn <= 256, got %d\n", n_topn);
        return;
    }
    
    // 计算内存大小（只分配最终输出，无需中间缓冲区）
    size_t size_query_group = n_query * n_dim * sizeof(float);
    size_t size_cluster_query_offset = (n_total_clusters + 1) * sizeof(int);  // 标准offset数组格式
    size_t size_cluster_query_data = n_query * n_cluster * sizeof(int);
    size_t size_cluster_vector_index = n_total_clusters * sizeof(int);
    size_t size_cluster_vector_num = n_total_clusters * sizeof(int);
    size_t size_cluster_vector = n_total_vectors * n_dim * sizeof(float);
    size_t size_topn_index = n_query * n_topn * sizeof(int);
    size_t size_topn_dist = n_query * n_topn * sizeof(float);
    
    // 分配设备内存（无需中间缓冲区）
    float *d_query_group, *d_cluster_vector, *d_topn_dist, *d_query_norm, *d_cluster_vector_norm;
    int *d_cluster_query_offset, *d_cluster_query_data;
    int *d_cluster_vector_index, *d_cluster_vector_num, *d_topn_index;
    
    dim3 clusterDim(n_total_vectors);
    dim3 vectorDim(n_dim);
    dim3 queryDim(n_query);
    
    // 计算L2范数
    {
        CUDATimer timer_compute("Memory Allocation", ENABLE_CUDA_TIMING);
        // GPU内存分配
        cudaMalloc(&d_query_group, size_query_group);
        cudaMalloc(&d_cluster_query_offset, size_cluster_query_offset);
        cudaMalloc(&d_cluster_query_data, size_cluster_query_data);
        cudaMalloc(&d_cluster_vector_index, size_cluster_vector_index);
        cudaMalloc(&d_cluster_vector_num, size_cluster_vector_num);
        cudaMalloc(&d_cluster_vector, size_cluster_vector);
        cudaMalloc(&d_query_norm, n_query * sizeof(float));  // 存储query的L2范数
        cudaMalloc(&d_cluster_vector_norm, n_total_vectors * sizeof(float));  // 存储cluster向量的L2范数
        cudaMalloc(&d_topn_index, size_topn_index);
        cudaMalloc(&d_topn_dist, size_topn_dist);
    }

    // 计算L2范数
    {
        CUDATimer timer_compute("Data Transfer: H2D", ENABLE_CUDA_TIMING);
        // 复制数据到设备内存
        cudaMemcpy(d_query_group, h_query_group[0], size_query_group, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cluster_query_offset, h_cluster_query_offset, size_cluster_query_offset, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cluster_query_data, h_cluster_query_data, size_cluster_query_data, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cluster_vector_index, h_cluster_vector_index, size_cluster_vector_index, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cluster_vector_num, h_cluster_vector_num, size_cluster_vector_num, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cluster_vector, h_cluster_vector[0], size_cluster_vector, cudaMemcpyHostToDevice);
    }

    // 计算L2范数
    {
        CUDATimer timer_compute("Kernel Execution: l2 Norm", ENABLE_CUDA_TIMING);
        // 计算查询向量的L2范数
        l2_norm_kernel<<<queryDim, vectorDim, n_dim * sizeof(float)>>>(
            d_query_group, d_query_norm, 
            n_query, n_dim
        );
        l2_norm_kernel<<<clusterDim, vectorDim, n_dim * sizeof(float)>>>(
            d_cluster_vector, d_cluster_vector_norm, 
            n_total_vectors, n_dim
        );
        cudaDeviceSynchronize();
        
        // // Debug: 打印GPU计算的norm值（小数据时）
        // if (n_query <= 4) {
        //     float* h_query_norm_debug = (float*)malloc(n_query * sizeof(float));
        //     float* h_cluster_vector_norm_debug = (float*)malloc(n_total_vectors * sizeof(float));
            
        //     // 复制所有数据进行对比和验证
        //     float* h_cluster_vector_all = (float*)malloc(n_total_vectors * n_dim * sizeof(float));
        //     cudaMemcpy(h_query_norm_debug, d_query_norm, n_query * sizeof(float), cudaMemcpyDeviceToHost);
        //     cudaMemcpy(h_cluster_vector_norm_debug, d_cluster_vector_norm, n_total_vectors * sizeof(float), cudaMemcpyDeviceToHost);
        //     cudaMemcpy(h_cluster_vector_all, d_cluster_vector, n_total_vectors * n_dim * sizeof(float), cudaMemcpyDeviceToHost);
            
        //     printf("[GPU Debug] Query norms: ");
        //     for (int q = 0; q < n_query; q++) {
        //         printf("Q%d=%.5f ", q, h_query_norm_debug[q]);
        //     }
        //     printf("\n");
            
        //     printf("[GPU Debug] Vector norms (first 10): ");
        //     int max_print = (n_total_vectors < 10) ? n_total_vectors : 10;
        //     for (int v = 0; v < max_print; v++) {
        //         // 手动计算GPU的norm用于验证
        //         float gpu_norm_squared = 0.0f;
        //         for (int d = 0; d < n_dim; d++) {
        //             float val = h_cluster_vector_all[v * n_dim + d];
        //             gpu_norm_squared += val * val;
        //         }
        //         printf("V%d=%.5f(calc=%.5f) ", v, h_cluster_vector_norm_debug[v], sqrtf(gpu_norm_squared));
        //     }
        //     printf("\n");
            
        //     printf("[GPU Debug] Vector 0 first 3 dims: %.5f %.5f %.5f\n", 
        //            h_cluster_vector_all[0], h_cluster_vector_all[1], h_cluster_vector_all[2]);
        //     if (n_total_vectors > 6) {
        //         printf("[GPU Debug] Vector 6 first 3 dims: %.5f %.5f %.5f\n", 
        //                h_cluster_vector_all[6 * n_dim], h_cluster_vector_all[6 * n_dim + 1], h_cluster_vector_all[6 * n_dim + 2]);
        //     }
            
        //     // 对比CPU和GPU的数据
        //     bool data_match = true;
        //     for (int v = 0; v < std::min(3, n_total_vectors); v++) {
        //         for (int d = 0; d < n_dim; d++) {
        //             float cpu_val = h_cluster_vector[v][d];
        //             float gpu_val = h_cluster_vector_all[v * n_dim + d];
        //             if (fabs(cpu_val - gpu_val) > 1e-5f) {
        //                 printf("[Data Mismatch] Vector %d dim %d: CPU=%.5f GPU=%.5f\n", v, d, cpu_val, gpu_val);
        //                 data_match = false;
        //             }
        //         }
        //     }
        //     if (data_match) {
        //         printf("[Data Check] CPU and GPU data match ✓\n");
        //     }
            
        //     free(h_query_norm_debug);
        //     free(h_cluster_vector_norm_debug);
        //     free(h_cluster_vector_all);
        // }
    }
    
    // 调用流式融合余弦距离top-k计算（v1版本）
    {
        CUDATimer timer_compute("Kernel Execution: Fusion Cos Top-K (v1)", ENABLE_CUDA_TIMING);
        
        cuda_cos_topk_warpsort_fine_v1(
            d_query_group, d_cluster_vector, d_cluster_query_offset, d_cluster_query_data,
            d_cluster_vector_index, d_cluster_vector_num,
            d_query_norm, d_cluster_vector_norm,
            d_topn_index, d_topn_dist,
            n_query, n_total_clusters, n_dim, n_topn, n_total_vectors
        );
        
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    }

    // 调用流式融合余弦距离top-k计算（v1版本）
    {
        CUDATimer timer_compute("Data Transfer: D2H", ENABLE_CUDA_TIMING);
        // 复制结果回主机内存
        cudaMemcpy(h_query_topn_index[0], d_topn_index, size_topn_index, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_query_topn_dist[0], d_topn_dist, size_topn_dist, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS;
    }

    {
        CUDATimer timer_compute("Memory Free", ENABLE_CUDA_TIMING);
        // 释放设备内存
        cudaFree(d_query_group);
        cudaFree(d_cluster_query_offset);
        cudaFree(d_cluster_query_data);
        cudaFree(d_cluster_vector_index);
        cudaFree(d_cluster_vector_num);
        cudaFree(d_cluster_vector);
        cudaFree(d_query_norm);
        cudaFree(d_cluster_vector_norm);
        cudaFree(d_topn_index);
        cudaFree(d_topn_dist);
    }
}


