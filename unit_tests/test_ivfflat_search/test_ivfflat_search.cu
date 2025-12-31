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
#include "../../cuda/ivf_search/ivf_search.cuh"
#include "../../cuda/dataset/dataset.cuh"
#include "../../cuda/utils.cuh"
#include "../common/test_utils.cuh"
#include "../common/cpu_search.h"

// ============================================================
// Test Configuration
// ============================================================
struct IVFFlatSearchCase {
    int n;              // 数据集大小
    int dim;            // 向量维度
    int n_clusters;              // cluster数量
    int n_query;        // 查询数量
    int n_probes;       // 粗筛选择的cluster数
    int topk;           // 最终输出的topk数量
    int kmeans_iters;   // n_clusters-means迭代次数
    bool use_minibatch; // 是否使用Minibatch算法
    DistanceType dist;   // 距离类型
};

// ============================================================
// Test Case Runner
// ============================================================
static std::vector<double> run_case(const IVFFlatSearchCase& cfg) {
    const int n = cfg.n, dim = cfg.dim, n_clusters = cfg.n_clusters;
    const int n_query = cfg.n_query, n_probes = cfg.n_probes, topk = cfg.topk;
    
    if (!QUIET) {
        COUT_ENDL("========================================");
        COUT_VAL("IVF-Flat Search Test: n=", cfg.n,
                " dim=", cfg.dim,
                " n_clusters=", cfg.n_clusters,
                " n_query=", cfg.n_query,
                " n_probes=", cfg.n_probes,
                " topk=", cfg.topk,
                " iters=", cfg.kmeans_iters,
                " dist=", (cfg.dist == L2_DISTANCE ? "L2" : "COSINE"),
                " algo=", (cfg.use_minibatch ? "MINIBATCH" : "LLOYD"));
        // COUT_ENDL("========================================");
    }
    
    // Step 1: 初始化ClusterDataset（使用K-means聚类）
    ClusterDataset dataset;
    float kmeans_objective = 0.0f;
    double kmeans_ms = 0.0;
    MEASURE_MS_AND_SAVE("IVF n_clusters-means耗时:", kmeans_ms,
        dataset.init_with_kmeans(
            n, dim, n_clusters,
            &kmeans_objective,  // h_objective
            cfg.kmeans_iters,
            cfg.use_minibatch,
            cfg.dist
        );
    );
    
    if (!QUIET) {
        COUT_VAL("K-means objective: ", kmeans_objective);
    }
    
    // Step 2: 生成查询向量（使用多线程随机初始化）
    float* h_query_batch = (float*)std::aligned_alloc(64, (size_t)n_query * dim * sizeof(float));
    init_array_multithreaded(h_query_batch, (size_t)n_query * dim, 5678, -1.0f, 1.0f);
    
    // Step 3: CPU参考搜索（使用GPU聚类的结果，确保CPU和GPU使用相同的聚类）
    // 注意：CPU搜索不包含聚类部分，直接使用dataset中已经聚类好的数据
    int** cpu_idx = (int**)malloc_vector_list(n_query, topk, sizeof(int));
    float** cpu_dist = (float**)malloc_vector_list(n_query, topk, sizeof(float));
    
    double cpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("CPU搜索耗时:", cpu_ms,
        cpu_coarse_fine_search(cfg.n_query, cfg.dim, cfg.n_clusters, cfg.n_probes, cfg.topk,
                              h_query_batch, 
                              dataset.reordered_data,    // 使用GPU聚类后重排的数据
                              dataset.centroids,         // 使用GPU聚类的中心
                              dataset.cluster_info,      // 使用GPU聚类的cluster信息
                              cfg.dist,
                              cpu_idx, cpu_dist);
    );
    
    // Step 4: GPU搜索（使用与CPU相同的GPU聚类结果）
    // 准备GPU数据
    float* d_query_batch = nullptr;
    int* d_cluster_size = nullptr;
    float* d_cluster_vectors = nullptr;
    float* d_cluster_centers = nullptr;
    float* d_topk_dist = nullptr;
    int* d_topk_index = nullptr;
    
    // 4.1 复制查询向量
    cudaMalloc(&d_query_batch, sizeof(float) * (size_t)n_query * dim);
    cudaMemcpy(d_query_batch, h_query_batch, sizeof(float) * (size_t)n_query * dim, cudaMemcpyHostToDevice);
    
    // 4.2 复制cluster sizes（从GPU聚类结果获取）
    cudaMalloc(&d_cluster_size, sizeof(int) * (size_t)n_clusters);
    cudaMemcpy(d_cluster_size, dataset.cluster_info.counts, sizeof(int) * (size_t)n_clusters, cudaMemcpyHostToDevice);
    
    // 4.3 复制重排后的向量数据（使用GPU聚类后重排的数据）
    size_t data_size = (size_t)n * dim;
    cudaMalloc(&d_cluster_vectors, sizeof(float) * data_size);
    cudaMemcpy(d_cluster_vectors, dataset.reordered_data, sizeof(float) * data_size, cudaMemcpyHostToDevice);
    
    // 4.4 复制聚类中心（使用GPU聚类的中心）
    cudaMalloc(&d_cluster_centers, sizeof(float) * (size_t)n_clusters * dim);
    cudaMemcpy(d_cluster_centers, dataset.centroids, sizeof(float) * (size_t)n_clusters * dim, cudaMemcpyHostToDevice);
    
    // 4.5 生成并复制 cluster 索引数组 [0, 1, 2, ..., n_clusters-1]
    int* d_initial_indices = nullptr;
    cudaMalloc(&d_initial_indices, sizeof(int) * (size_t)n_query * n_clusters);
    CHECK_CUDA_ERRORS;
    
    // 使用kernel在GPU上生成顺序索引
    dim3 block(256);
    dim3 grid((n_query * n_clusters + block.x - 1) / block.x);
    generate_sequential_indices_kernel<<<grid, block>>>(
        d_initial_indices, n_query, n_clusters);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    
    // 4.6 分配输出缓冲区
    cudaMalloc(&d_topk_dist, sizeof(float) * (size_t)n_query * topk);
    cudaMalloc(&d_topk_index, sizeof(int) * (size_t)n_query * topk);
    
    // 4.7 运行GPU搜索（使用与CPU相同的GPU聚类结果）
    int** gpu_idx = (int**)malloc_vector_list(n_query, topk, sizeof(int));
    float** gpu_dist = (float**)malloc_vector_list(n_query, topk, sizeof(float));
    
    double gpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("GPU搜索耗时:", gpu_ms,
        batch_search_pipeline(
            d_query_batch,
            d_cluster_size,
            d_cluster_vectors,
            d_cluster_centers,
            d_initial_indices,  // 使用生成的 cluster 索引数组
            d_topk_dist,
            d_topk_index,
            n_query,
            dim,
            n_clusters,
            n,
            n_probes,
            topk,
            cfg.dist
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );
    
    // Step 5: 复制结果回host（GPU返回的是重排后的位置索引）
    int* h_gpu_idx_raw = (int*)std::malloc(sizeof(int) * (size_t)n_query * topk);
    cudaMemcpy(gpu_dist[0], d_topk_dist, sizeof(float) * (size_t)n_query * topk, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_idx_raw, d_topk_index, sizeof(int) * (size_t)n_query * topk, cudaMemcpyDeviceToHost);
    
    // Step 6: 将重排后的位置索引映射回原始索引
    for (int qi = 0; qi < n_query; ++qi) {
        for (int ki = 0; ki < topk; ++ki) {
            int reordered_pos = h_gpu_idx_raw[qi * topk + ki];
            if (reordered_pos >= 0 && reordered_pos < n) {
                gpu_idx[qi][ki] = dataset.reordered_indices[reordered_pos];
            } else {
                gpu_idx[qi][ki] = -1;  // 无效索引
            }
        }
    }
    std::free(h_gpu_idx_raw);
    
    // Step 7: 验证结果（CPU和GPU使用相同的聚类，应该得到相同的结果）
    bool pass = (cfg.dist == L2_DISTANCE) 
        ? compare_set_2D_relative<float>(cpu_dist, gpu_dist, n_query, topk, 1e-3f)
        : compare_set_2D<float>(cpu_dist, gpu_dist, n_query, topk, 1e-5f);
    
    // 验证索引是否匹配（允许距离相同的情况）
    int mismatch_count = 0;
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
                        mismatch_count++;
                        if (!QUIET && mismatch_count <= 3) {
                            fprintf(stderr, "WARNING: Query %d, Top-%d: CPU idx=%d, GPU idx=%d, CPU dist=%.6f, GPU dist=%.6f\n",
                                   qi, ki, cpu_idx_val, gpu_idx_val, cpu_dist_val, gpu_dist_val);
                        }
                    }
                }
            }
        }
    }
    
    double pass_rate = pass ? 1.0 : 0.0;
    double speedup = (gpu_ms > 1e-6) ? (cpu_ms / gpu_ms) : 0.0;
    
    if (!QUIET) {
        COUT_ENDL("----- IVF-Flat Search Verify -----");
        COUT_VAL("pass=", (pass ? 1 : 0), " kmeans_ms=", kmeans_ms, " cpu_ms=", cpu_ms, " gpu_ms=", gpu_ms);
        COUT_VAL("speedup=", speedup, " mismatch_count=", mismatch_count);
        // COUT_ENDL("-----------------------------------");
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
    cudaFree(d_initial_indices);
    cudaFree(d_topk_dist);
    cudaFree(d_topk_index);
    
    std::free(h_query_batch);
    dataset.release();
    
    return {pass_rate, kmeans_ms, cpu_ms, gpu_ms, speedup, (double)mismatch_count, (double)kmeans_objective};
}

// ============================================================
// Main (Metrics table like test_kmeans)
// ============================================================
#ifdef __CUDACC__
__host__
#endif
int main(int argc, char** argv) {
    MetricsCollector metrics;
    metrics.set_columns("pass_rate", "n", "dim", "n_clusters", "n_query", "n_probes", "topk", 
                        "iters", "dist", "algo",
                        "kmeans_ms", "cpu_ms", "gpu_ms", "speedup", "mismatch_count", "kmeans_obj");
    metrics.set_num_repeats(1);
    
    cudaSetDevice(0);
    CHECK_CUDA_ERRORS;
    
    // 使用 PARAM_3D 组合测试 cfg.n, cfg.n_query, cfg.dim
    PARAM_3D(n, (10000, 20003, 100000, 240000, 1000000, 2000000),
             n_query, (1,20,34,100,10000),
             dim, (20,32,64,96, 115, 128, 192, 200, 256))
    // PARAM_3D(n, (10000, 1000000),
    //     n_query, (100),
    //     dim, (96))
    {
        IVFFlatSearchCase cfg;
        cfg.n = n;
        cfg.n_query = n_query;
        cfg.dim = dim;
        // cfg.n_clusters = (int)std::round(std::sqrt(cfg.n)); // 策略1
        cfg.n_clusters = (int)std::round(std::pow((double)0.1 * cfg.n, 2.0 / 3.0)); // 策略2
        cfg.n_probes = 10;
        cfg.topk = 10;
        cfg.kmeans_iters = 20;
        cfg.use_minibatch = true;
        cfg.dist = L2_DISTANCE;
        
        auto row = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto r = run_case(cfg);
            return {
                r[0],                         // pass_rate
                (double)cfg.n,
                (double)cfg.dim,
                (double)cfg.n_clusters,
                (double)cfg.n_query,
                (double)cfg.n_probes,
                (double)cfg.topk,
                (double)cfg.kmeans_iters,
                (double)cfg.dist,
                (double)(cfg.use_minibatch ? 1 : 0),  // algo
                r[1],                         // kmeans_ms
                r[2],                         // cpu_ms
                r[3],                         // gpu_ms
                r[4],                         // speedup
                r[5],                         // mismatch_count
                r[6],                         // kmeans_obj
            };
        });
    }
    
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    
    metrics.print_table();
    metrics.export_csv("ivfflat_search_metrics.csv");
    COUT_ENDL("IVF-Flat Search tests completed successfully!");
    return 0;
}

