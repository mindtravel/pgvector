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
#include <map>
#include <set>

#include "../../cuda/pch.h"
#include "../../cuda/ivf_search/ivf_search.cuh"
#include "../../cuda/dataset/dataset.cuh"
#include "../../cuda/utils.cuh"
#include "../common/test_utils.cuh"
#include "../cpu_utils/cpu_utils.h"

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
    }
    
    // Step 1: 初始化ClusterDataset（使用K-means聚类）
    ClusterDataset dataset;
    float kmeans_objective = 0.0f;
    double kmeans_ms = 0.0;
    MEASURE_MS_AND_SAVE("IVF n_clusters-means耗时:", kmeans_ms,
        dataset.init_with_kmeans(
            n, dim, n_clusters,
            &kmeans_objective,  // h_objective
            nullptr,
            cfg.kmeans_iters,
            cfg.use_minibatch,
            cfg.dist
        );
    );
    
    // if (!QUIET) {
    //     COUT_VAL("K-means objective: ", kmeans_objective);
    // }
    COUT_ENDL("K-means completed successfully!");
    
    // Step 2: 生成查询向量（使用多线程随机初始化）
    float* h_query_batch = (float*)std::aligned_alloc(64, (size_t)n_query * dim * sizeof(float));
    init_array_multithreaded(h_query_batch, (size_t)n_query * dim, 5678, -1.0f, 1.0f);
    
    // Step 3: CPU参考搜索（使用GPU聚类的结果，确保CPU和GPU使用相同的聚类）
    // 注意：CPU搜索不包含聚类部分，直接使用dataset中已经聚类好的数据
    int** cpu_idx = malloc_vector_list<int>(n_query, topk);
    float** cpu_dist = malloc_vector_list<float>(n_query, topk);
    
    // 分配粗筛结果数组
    int** cpu_coarse_idx = malloc_vector_list<int>(n_query, n_probes);
    float** cpu_coarse_dist = malloc_vector_list<float>(n_query, n_probes);
    
    double cpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("CPU搜索耗时:", cpu_ms,
        cpu_coarse_fine_search_lookup(cfg.n_query, cfg.dim, cfg.n_clusters, cfg.n_probes, cfg.topk,
                              h_query_batch, 
                              dataset.reordered_data,    // 使用GPU聚类后重排的数据
                              dataset.centroids,         // 使用GPU聚类的中心
                              dataset.cluster_info,      // 使用GPU聚类的cluster信息
                              cfg.dist,
                              cpu_idx, cpu_dist,
                              cpu_coarse_idx, cpu_coarse_dist,  // 粗筛结果
                              dataset.reordered_indices);  // 回表映射数组
    );
    
    // Step 4: GPU搜索（使用与CPU相同的GPU聚类结果）
    // 准备GPU数据
    float* d_query_batch = nullptr;
    int* d_cluster_size = nullptr;
    float* d_cluster_vectors = nullptr;
    float* d_cluster_centers = nullptr;
    int* d_initial_indices = nullptr;
    float* d_topk_dist = nullptr;
    int* d_topk_index = nullptr;
    int* d_reordered_indices = nullptr;
    
    // 分配GPU内存
    cudaMalloc(&d_query_batch, sizeof(float) * (size_t)n_query * dim);
    cudaMalloc(&d_cluster_size, sizeof(int) * (size_t)n_clusters);
    cudaMalloc(&d_cluster_vectors, sizeof(float) * (size_t)n * dim);
    cudaMalloc(&d_cluster_centers, sizeof(float) * (size_t)n_clusters * dim);
    cudaMalloc(&d_initial_indices, sizeof(int) * (size_t)n_query * n_clusters);
    cudaMalloc(&d_topk_dist, sizeof(float) * (size_t)n_query * topk);
    cudaMalloc(&d_topk_index, sizeof(int) * (size_t)n_query * topk);
    cudaMalloc(&d_reordered_indices, sizeof(int) * (size_t)n);
    
    // 复制数据到GPU
    cudaMemcpy(d_query_batch, h_query_batch, sizeof(float) * (size_t)n_query * dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_size, dataset.cluster_info.counts, sizeof(int) * (size_t)n_clusters, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vectors, dataset.reordered_data, sizeof(float) * (size_t)n * dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_centers, dataset.centroids, sizeof(float) * (size_t)n_clusters * dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_reordered_indices, dataset.reordered_indices, sizeof(int) * (size_t)n, cudaMemcpyHostToDevice);
    
    // 生成并复制 cluster 索引数组 [0, 1, 2, ..., n_clusters-1]
    dim3 block(256);
    dim3 grid((n_query * n_clusters + block.x - 1) / block.x);
    generate_sequential_indices_kernel<<<grid, block>>>(
        d_initial_indices, n_query, n_clusters);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    
    // 分配GPU结果数组（host端）
    int** gpu_idx = malloc_vector_list<int>(n_query, topk);
    float** gpu_dist = malloc_vector_list<float>(n_query, topk);
    int** h_gpu_coarse_idx = malloc_vector_list<int>(n_query, n_probes);
    float** h_gpu_coarse_dist = malloc_vector_list<float>(n_query, n_probes);
    
    double gpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("GPU搜索耗时:", gpu_ms,
        ivf_search_lookup(
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
            cfg.dist,
            h_gpu_coarse_idx,   // 粗筛结果索引
            h_gpu_coarse_dist,  // 粗筛结果距离
            d_reordered_indices // 回表映射数组
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );
    
    // Step 5: 复制结果回host（GPU返回原始索引，CPU也需要返回原始索引）
    cudaMemcpy(gpu_dist[0], d_topk_dist, sizeof(float) * (size_t)n_query * topk, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_idx[0], d_topk_index, sizeof(int) * (size_t)n_query * topk, cudaMemcpyDeviceToHost);
    
    // Step 7: 验证粗筛结果（只比较距离数组）
    std::vector<MismatchInfo> coarse_mismatches;
    coarse_mismatches = (cfg.dist == L2_DISTANCE)   ? 
        compare_set_2D_with_mismatches_relative<float>(cpu_coarse_dist, h_gpu_coarse_dist, cpu_coarse_idx, h_gpu_coarse_idx, n_query, n_probes, 1e-3f)
        : compare_set_2D_with_mismatches<float>(cpu_coarse_dist, h_gpu_coarse_dist, cpu_coarse_idx, h_gpu_coarse_idx, n_query, n_probes, 1e-5f);
    bool coarse_pass = coarse_mismatches.empty();
    int coarse_mismatch_count = coarse_mismatches.size();
    
    // Step 8: 验证精筛结果（使用智能比较函数记录不匹配信息）
    std::vector<MismatchInfo> fine_mismatches;
    fine_mismatches = (cfg.dist == L2_DISTANCE) ? 
        compare_set_2D_with_mismatches_relative<float>(cpu_dist, gpu_dist, cpu_idx, gpu_idx, n_query, topk, 1e-3f)
        : compare_set_2D_with_mismatches<float>(cpu_dist, gpu_dist, cpu_idx, gpu_idx, n_query, topk, 1e-5f);
    bool pass = fine_mismatches.empty();
    int fine_mismatch_count = fine_mismatches.size();
    
    // 总体通过需要粗筛和精筛都通过
    bool overall_pass = coarse_pass && pass;
    
    double pass_rate = overall_pass ? 1.0 : 0.0;
    double speedup = (gpu_ms > 1e-6) ? (cpu_ms / gpu_ms) : 0.0;
    
    if (!QUIET) {
        COUT_ENDL("----- IVF-Flat Search Lookup Verify -----");
        COUT_VAL("coarse_pass=", (coarse_pass ? 1 : 0), " fine_pass=", (pass ? 1 : 0), " overall_pass=", (overall_pass ? 1 : 0));
        COUT_VAL("kmeans_ms=", kmeans_ms, " cpu_ms=", cpu_ms, " gpu_ms=", gpu_ms, "speedup=", speedup);
        COUT_VAL(" coarse_mismatch_count=", coarse_mismatch_count, " fine_mismatch_count=", fine_mismatch_count);
        
    }
    
    // Cleanup
    free_vector_list((void**)cpu_idx);
    free_vector_list((void**)gpu_idx);
    free_vector_list((void**)cpu_dist);
    free_vector_list((void**)gpu_dist);
    free_vector_list((void**)cpu_coarse_idx);
    free_vector_list((void**)h_gpu_coarse_idx);
    free_vector_list((void**)cpu_coarse_dist);
    free_vector_list((void**)h_gpu_coarse_dist);
    
    cudaFree(d_query_batch);
    cudaFree(d_cluster_size);
    cudaFree(d_cluster_vectors);
    cudaFree(d_cluster_centers);
    cudaFree(d_initial_indices);
    cudaFree(d_topk_dist);
    cudaFree(d_topk_index);
    cudaFree(d_reordered_indices);
    
    std::free(h_query_batch);
    dataset.release();
    
    return {pass_rate, kmeans_ms, cpu_ms, gpu_ms, speedup, (double)coarse_mismatch_count, (double)fine_mismatch_count, (double)kmeans_objective};
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
                        "kmeans_ms", "cpu_ms", "gpu_ms", "speedup", "coarse_mismatch_count", "fine_mismatch_count", "kmeans_obj");
    metrics.set_num_repeats(1);
    
    cudaSetDevice(0);
    CHECK_CUDA_ERRORS;
    
    // 使用 PARAM_3D 组合测试 cfg.n, cfg.n_query, cfg.dim
    PARAM_3D(n, (10000, 20003, 100000, 240000, 1000000, 2000000),
    // PARAM_3D(n, (5000000),
    // PARAM_3D(n, (10000),
             n_query, (10000),
             dim, (96, 100, 128, 200))
            //  dim, (20,32,64,96, 115, 128, 192, 200, 256))
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
                r[5],                         // coarse_mismatch_count
                r[6],                         // fine_mismatch_count
                r[7],                         // kmeans_obj
            };
        });
    }
    
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    
    metrics.print_table();
    metrics.export_csv("ivfflat_search_lookup_metrics.csv");
    COUT_ENDL("IVF-Flat Search tests completed successfully!");
    return 0;
}

