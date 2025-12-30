#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../../cuda/pch.h"
#include "../../cuda/kmeans/kmeans.cuh"
#include "../../cuda/integrate_screen/integrate_screen.cuh"
#include "../common/test_utils.cuh"
#include "../common/cpu_search.h"

// ============================================================
// Test Configuration
// ============================================================
struct IVFFlatSearchCase {
    int n;              // 数据集大小
    int dim;            // 向量维度
    int k;              // cluster数量
    int n_query;        // 查询数量
    int n_probes;       // 粗筛选择的cluster数
    int topk;           // 最终输出的topk数量
    int kmeans_iters;   // K-means迭代次数
    bool use_minibatch; // 是否使用Minibatch算法
    DistanceType dist;   // 距离类型
};

// ============================================================
// Test Runner
// ============================================================
static bool run_test(const IVFFlatSearchCase& cfg) {
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;
    const int n_query = cfg.n_query, n_probes = cfg.n_probes, topk = cfg.topk;
    
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "IVF-Flat Search Test:\n");
    fprintf(stderr, "  n=%d, dim=%d, k=%d\n", n, dim, k);
    fprintf(stderr, "  n_query=%d, n_probes=%d, topk=%d\n", n_query, n_probes, topk);
    fprintf(stderr, "  kmeans_iters=%d, algo=%s\n", cfg.kmeans_iters, cfg.use_minibatch ? "MINIBATCH" : "LLOYD");
    fprintf(stderr, "========================================\n");
    
    // Step 1: 生成测试数据
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    size_t data_size = (size_t)n * dim;
    float* h_data = (float*)std::aligned_alloc(64, data_size * sizeof(float));
    if (!h_data) {
        fprintf(stderr, "ERROR: Failed to allocate h_data\n");
        return false;
    }
    
    for (size_t i = 0; i < data_size; ++i) {
        h_data[i] = dist(rng);
    }
    
    // Step 2: 初始化聚类中心
    float* h_init_centroids = nullptr;
    cudaMallocHost(&h_init_centroids, sizeof(float) * (size_t)k * dim);
    KMeansCase kmeans_cfg;
    kmeans_cfg.n = n;
    kmeans_cfg.dim = dim;
    kmeans_cfg.k = k;
    kmeans_cfg.iters = cfg.kmeans_iters;
    kmeans_cfg.minibatch_iters = cfg.kmeans_iters * 4; // Minibatch需要更多迭代
    kmeans_cfg.seed = 1234;
    kmeans_cfg.dist = cfg.dist;
    kmeans_cfg.dtype = USE_FP32;
    
    init_centroids_by_sampling(kmeans_cfg, h_data, h_init_centroids);
    
    // Step 3: 分配GPU内存
    float* d_centroids = nullptr;
    cudaMalloc(&d_centroids, sizeof(float) * (size_t)k * dim);
    cudaMemcpy(d_centroids, h_init_centroids, sizeof(float) * (size_t)k * dim, cudaMemcpyHostToDevice);
    
    // Step 4: 分配重排后的数据缓冲区
    float* h_data_reordered = (float*)std::aligned_alloc(64, data_size * sizeof(float));
    if (!h_data_reordered) {
        fprintf(stderr, "ERROR: Failed to allocate h_data_reordered\n");
        std::free(h_data);
        cudaFree(d_centroids);
        cudaFreeHost(h_init_centroids);
        return false;
    }
    
    // Step 5: 运行IVF K-means（聚类 + 重排）
    ClusterInfo cluster_info;
    float kmeans_objective = 0.0f;
    const int batch_size = 1 << 20; // 1M per batch
    
    double kmeans_ms = 0.0;
    MEASURE_MS_AND_SAVE("IVF K-means耗时:", kmeans_ms,
        bool success = ivf_kmeans(kmeans_cfg, h_data, h_data_reordered, d_centroids,
                                 &cluster_info, cfg.use_minibatch, 0, batch_size, &kmeans_objective);
        if (!success) {
            fprintf(stderr, "ERROR: ivf_kmeans failed\n");
            return false;
        }
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );
    
    fprintf(stderr, "K-means objective: %f\n", kmeans_objective);
    
    // Step 6: 拷贝最终的centroids回host（用于CPU参考搜索）
    float* h_centroids = (float*)std::aligned_alloc(64, sizeof(float) * (size_t)k * dim);
    cudaMemcpy(h_centroids, d_centroids, sizeof(float) * (size_t)k * dim, cudaMemcpyDeviceToHost);
    
    // Step 7: 生成查询向量
    float* h_query_batch = (float*)std::aligned_alloc(64, (size_t)n_query * dim * sizeof(float));
    for (int i = 0; i < n_query * dim; ++i) {
        h_query_batch[i] = dist(rng);
    }
    
    // Step 8: CPU参考搜索
    int** cpu_idx = (int**)malloc_vector_list(n_query, topk, sizeof(int));
    float** cpu_dist = (float**)malloc_vector_list(n_query, topk, sizeof(float));
    
    double cpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("CPU搜索耗时:", cpu_ms,
        cpu_coarse_fine_search(cfg.n_query, cfg.dim, cfg.k, cfg.n_probes, cfg.topk,
                              h_query_batch, h_data_reordered, 
                              h_centroids, cluster_info, cfg.dist,
                              cpu_idx, cpu_dist);
    );
    
    // Step 9: GPU搜索
    // 准备GPU数据
    float* d_query_batch = nullptr;
    int* d_cluster_size = nullptr;
    float* d_cluster_vectors = nullptr;
    float* d_cluster_centers = nullptr;
    float* d_topk_dist = nullptr;
    int* d_topk_index = nullptr;
    
    // 8.1 复制查询向量
    cudaMalloc(&d_query_batch, sizeof(float) * (size_t)n_query * dim);
    cudaMemcpy(d_query_batch, h_query_batch, sizeof(float) * (size_t)n_query * dim, cudaMemcpyHostToDevice);
    
    // 8.2 复制cluster sizes
    cudaMalloc(&d_cluster_size, sizeof(int) * (size_t)k);
    std::vector<int> h_cluster_sizes(k);
    for (int c = 0; c < k; ++c) {
        h_cluster_sizes[c] = cluster_info.counts[c];
    }
    cudaMemcpy(d_cluster_size, h_cluster_sizes.data(), sizeof(int) * (size_t)k, cudaMemcpyHostToDevice);
    
    // 8.3 复制重排后的向量数据（已经是连续存储）
    cudaMalloc(&d_cluster_vectors, sizeof(float) * data_size);
    cudaMemcpy(d_cluster_vectors, h_data_reordered, sizeof(float) * data_size, cudaMemcpyHostToDevice);
    
    // 8.4 复制聚类中心
    cudaMalloc(&d_cluster_centers, sizeof(float) * (size_t)k * dim);
    cudaMemcpy(d_cluster_centers, d_centroids, sizeof(float) * (size_t)k * dim, cudaMemcpyDeviceToDevice);
    
    // 8.5 分配输出缓冲区
    cudaMalloc(&d_topk_dist, sizeof(float) * (size_t)n_query * topk);
    cudaMalloc(&d_topk_index, sizeof(int) * (size_t)n_query * topk);
    
    // 8.6 运行GPU搜索
    int** gpu_idx = (int**)malloc_vector_list(n_query, topk, sizeof(int));
    float** gpu_dist = (float**)malloc_vector_list(n_query, topk, sizeof(float));
    
    double gpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("GPU搜索耗时:", gpu_ms,
        batch_search_pipeline(
            d_query_batch,
            d_cluster_size,
            d_cluster_vectors,
            d_cluster_centers,
            nullptr,  // d_initial_indices: nullptr表示内部生成顺序索引
            d_topk_dist,
            d_topk_index,
            n_query,
            dim,
            k,
            n,
            n_probes,
            topk,
            cfg.dist
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );
    
    // 8.7 复制结果回host
    cudaMemcpy(gpu_dist[0], d_topk_dist, sizeof(float) * (size_t)n_query * topk, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_idx[0], d_topk_index, sizeof(int) * (size_t)n_query * topk, cudaMemcpyDeviceToHost);
    
    // Step 10: 验证结果
    bool pass = (cfg.dist == L2_DISTANCE) 
        ? compare_set_2D_relative<float>(cpu_dist, gpu_dist, n_query, topk, 1e-4f)
        : compare_set_2D<float>(cpu_dist, gpu_dist, n_query, topk, 1e-5f);
    
    // 验证索引是否匹配（允许距离相同的情况）
    if (pass) {
        for (int qi = 0; qi < n_query && qi < 5; ++qi) {
            for (int ki = 0; ki < topk; ++ki) {
                int cpu_idx_val = cpu_idx[qi][ki];
                int gpu_idx_val = gpu_idx[qi][ki];
                if (cpu_idx_val != gpu_idx_val) {
                    // 检查距离是否相同（允许索引不同但距离相同的情况）
                    float cpu_dist_val = cpu_dist[qi][ki];
                    float gpu_dist_val = gpu_dist[qi][ki];
                    float rel_err = std::abs(cpu_dist_val - gpu_dist_val) / std::max(1e-6f, std::abs(cpu_dist_val));
                    if (rel_err > 1e-4f) {
                        fprintf(stderr, "WARNING: Query %d, Top-%d: CPU idx=%d, GPU idx=%d, CPU dist=%.6f, GPU dist=%.6f\n",
                               qi, ki, cpu_idx_val, gpu_idx_val, cpu_dist_val, gpu_dist_val);
                    }
                }
            }
        }
    }
    
    fprintf(stderr, "\n结果验证: %s\n", pass ? "PASS" : "FAIL");
    fprintf(stderr, "K-means耗时: %.2f ms\n", kmeans_ms);
    fprintf(stderr, "CPU搜索耗时: %.2f ms\n", cpu_ms);
    fprintf(stderr, "GPU搜索耗时: %.2f ms\n", gpu_ms);
    if (gpu_ms > 1e-6) {
        fprintf(stderr, "搜索加速比: %.2fx\n", cpu_ms / gpu_ms);
    }
    
    // Cleanup
    free_vector_list((void**)cpu_idx);
    free_vector_list((void**)gpu_idx);
    free_vector_list((void**)cpu_dist);
    free_vector_list((void**)gpu_dist);
    
    cudaFree(d_query_batch);
    cudaFree(d_cluster_size);
    cudaFree(d_cluster_vectors);
    cudaFree(d_cluster_centers);
    cudaFree(d_topk_dist);
    cudaFree(d_topk_index);
    cudaFree(d_centroids);
    
    std::free(h_data);
    std::free(h_data_reordered);
    std::free(h_query_batch);
    std::free(h_centroids);
    cudaFreeHost(h_init_centroids);
    free_cluster_info(&cluster_info, false);
    
    return pass;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    // 默认测试配置
    IVFFlatSearchCase cfg;
    cfg.n = 10000;
    cfg.dim = 96;
    cfg.k = 100;
    cfg.n_query = 100;
    cfg.n_probes = 10;
    cfg.topk = 10;
    cfg.kmeans_iters = 5;
    cfg.use_minibatch = false;
    cfg.dist = L2_DISTANCE;
    
    // 解析命令行参数
    if (argc >= 2) cfg.n = std::atoi(argv[1]);
    if (argc >= 3) cfg.dim = std::atoi(argv[2]);
    if (argc >= 4) cfg.k = std::atoi(argv[3]);
    if (argc >= 5) cfg.n_query = std::atoi(argv[4]);
    if (argc >= 6) cfg.n_probes = std::atoi(argv[5]);
    if (argc >= 7) cfg.topk = std::atoi(argv[6]);
    if (argc >= 8) cfg.use_minibatch = (std::atoi(argv[7]) != 0);
    
    cudaSetDevice(0);
    CHECK_CUDA_ERRORS;
    
    bool pass = run_test(cfg);
    
    return pass ? 0 : 1;
}

