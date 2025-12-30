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
#include "../common/test_utils.cuh"
#include "../common/cpu_kmeans.h"

// ============================================================
// Algorithm Version Enum
// ============================================================
enum KMeansAlgorithm {
    KMEANS_LLOYD = 0,      // 全量 Lloyd 算法
    KMEANS_MINIBATCH = 1   // Minibatch 算法
};

// ============================================================
// Utilities (CPU)
// ============================================================
// cpu_kmeans_lloyd 已移至 ../common/cpu_kmeans.h

// ============================================================
// Test Case Runner
// ============================================================
static std::vector<double> run_case(const KMeansCase& cfg, float* h_pool_data, KMeansAlgorithm algo) {
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;


    // 直接使用池数据（不进行归一化）
    float* h_data = h_pool_data;

    // 初始化聚类中心：使用统一的初始化函数确保CPU和GPU使用相同的起始聚类中心
    // 该函数使用 cfg.seed 确保确定性初始化（使用 pinned memory）
    float* h_init_centroids = nullptr;
    cudaMallocHost(&h_init_centroids, sizeof(float) * (size_t)k * dim);
    init_centroids_by_sampling(cfg, h_data, h_init_centroids);

    // CPU reference - 使用与GPU相同的初始化聚类中心
    std::vector<int> h_assign_cpu(n);
    std::vector<float> h_centroids_cpu((size_t)k * dim);
    std::memcpy(h_centroids_cpu.data(), h_init_centroids, sizeof(float) * (size_t)k * dim);
    float cpu_obj = 0.0f;

    double cpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("cpu_kmeans耗时:", cpu_ms,
        // cpu_kmeans_lloyd(cfg,
        //                  h_data,
        //                  h_assign_cpu.data(),
        //                  h_centroids_cpu.data(),
        //                  &cpu_obj);
    );

    // GPU buffers - 使用与CPU相同的初始化聚类中心
    float* d_centroids = nullptr;
    cudaMalloc(&d_centroids, sizeof(float) * (size_t)k * dim);
    cudaMemcpyAsync(d_centroids, h_init_centroids, sizeof(float) * (size_t)k * dim, 
                   cudaMemcpyHostToDevice);

    // 分配重排后的向量输出缓冲区（pageable memory）
    size_t reordered_size = (size_t)n * (size_t)dim;
    size_t reordered_bytes = reordered_size * sizeof(float);
    float* h_data_reordered = (float*)std::aligned_alloc(64, reordered_bytes);
    if (!h_data_reordered) {
        fprintf(stderr, "[test] ERROR: Failed to alloc h_data_reordered (%.2f GB)\n",
               reordered_bytes / 1024.0 / 1024.0 / 1024.0);
        cudaFree(d_centroids);
        cudaFreeHost(h_init_centroids);
        return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }

    // 使用 ivf_kmeans 完成整个流程：K-means聚类 + 构建permutation + 重排向量
    float gpu_obj = 0.0f;
    ClusterInfo h_cluster_info;
    bool use_minibatch = (algo == KMEANS_MINIBATCH);
    const int BATCH_SIZE = 1 << 20;  // 1M per batch
    
    double gpu_ms = 0.0;
    double reorder_ms = 0.0;
    bool reorder_success = false;
    
    MEASURE_MS_AND_SAVE("ivf_kmeans总耗时:", gpu_ms,
        reorder_success = ivf_kmeans(cfg, h_data, h_data_reordered, d_centroids,
                                     &h_cluster_info, use_minibatch, 0, BATCH_SIZE, &gpu_obj);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );
    
    // Copy GPU centroids back for verification
    std::vector<float> h_centroids_gpu((size_t)k * dim);
    cudaMemcpy(h_centroids_gpu.data(), d_centroids, sizeof(float) * (size_t)k * dim, cudaMemcpyDeviceToHost);
    
    // 验证cluster信息
    bool ok_info = true;
    if (h_cluster_info.offsets && h_cluster_info.counts && h_cluster_info.k > 0) {
        long long total = 0;
        for (int c = 0; c < h_cluster_info.k; ++c) {
            int cnt = h_cluster_info.counts[c];
            if (cnt < 0 || cnt > n || h_cluster_info.offsets[c] != total) {
                ok_info = false;
                break;
            }
            total += (long long)cnt;
        }
        if (total != (long long)n) ok_info = false;
    } else {
        ok_info = false;
    }
    
    reorder_success = reorder_success && ok_info;
    
    if (!QUIET) {
        if (reorder_success) {
            fprintf(stderr, "[test] IVF K-means completed successfully. "
                   "Sample: out[0..3] first dim0: %f %f %f %f\n",
                   h_data_reordered[0], h_data_reordered[(size_t)dim], 
                   h_data_reordered[(size_t)2 * dim], h_data_reordered[(size_t)3 * dim]);
        } else {
            fprintf(stderr, "[test] IVF K-means FAILED!\n");
        }
    }
    
    // Cleanup
    std::free(h_data_reordered);
    free_cluster_info(&h_cluster_info, false);
    cudaFree(d_centroids);
    cudaFreeHost(h_init_centroids);
    
    CHECK_CUDA_ERRORS;

    // verify: compare objective (relative) + centroid L2 error
    // 注意：FP16 + atomic 累加会引入误差，所以阈值稍放宽
    double rel_obj_err = std::abs((double)gpu_obj - (double)cpu_obj) / std::max(1.0, (double)cpu_obj);

    double centroid_rmse = 0.0;
    {
        float s = l2_distance_squared(h_centroids_cpu.data(), h_centroids_gpu.data(), h_centroids_cpu.size());
        centroid_rmse = std::sqrt(s / std::max<size_t>(1, h_centroids_cpu.size()));
    }

    double obj_tol = 1e-4;
    bool pass = rel_obj_err < obj_tol && reorder_success;

    double pass_rate = pass ? 1.0 : 0.0;
    double speedup = (gpu_ms > 1e-6) ? (cpu_ms / gpu_ms) : 0.0;
    reorder_ms = gpu_ms;  // ivf_kmeans包含了整个流程的时间

    if (!QUIET) {
        COUT_ENDL("----- KMeans Verify -----");
        COUT_VAL("cpu_obj=", cpu_obj, " gpu_obj=", gpu_obj, " rel_obj_err=", rel_obj_err);
        COUT_VAL("centroid_rmse=", centroid_rmse, " reorder_success=", (reorder_success ? 1 : 0), " pass=", (pass ? 1 : 0));
        COUT_ENDL("-------------------------");
    }

    return {pass_rate, gpu_ms, cpu_ms, speedup, (double)rel_obj_err, (double)centroid_rmse, (double)gpu_obj, (double)cpu_obj, reorder_ms, (double)reorder_success};
}

// ============================================================
// Main (Metrics table like your template)
// ============================================================
// 标记为仅主机代码，避免 CUDA 编译器处理 std::function
#ifdef __CUDACC__
__host__
#endif
int main(int argc, char** argv) {
    MetricsCollector metrics;
    metrics.set_columns("pass_rate", "n", "k", "dim", "iters", "dist", "algo",
                        "gpu_ms", "cpu_ms", "speedup", "rel_obj_err", "centroid_rmse",
                        "gpu_obj", "cpu_obj");
    metrics.set_num_repeats(1);

    // PARAM_3D(n, (10000, 20000),
    //  dim, (4,10,96,128,200),
    PARAM_1D(dim, (96)){
        // 在每个维度测试开始前，创建足够大的pinned memory池
        // 计算最大需要的池大小：max(n) * dim
        // 当前最大 n = 100000000，所以需要至少 100000000 * dim 个 floats
        // const size_t MAX_N = 1000000000;
        const size_t MAX_N = 500000000;
        const size_t POOL_SIZE = MAX_N * dim;
        float* h_pool_data = nullptr;
        cudaMallocHost(&h_pool_data, sizeof(float) * POOL_SIZE);
        init_array_multithreaded(h_pool_data, POOL_SIZE, 1234, -1.0f, 1.0f);
        if (!QUIET) {
            COUT_ENDL("========================================");
            COUT_VAL("Created pinned memory pool: ", POOL_SIZE, " floats (", 
                    (double)POOL_SIZE * sizeof(float) / (1024.0 * 1024.0 * 1024.0), " GB)");
            COUT_ENDL("========================================");
        }
        
        PARAM_3D(n, (20000, 1000000, 10000000, 100000000, 500000000), 
        // PARAM_3D(n, (20000, 100000, 1000000, 50000000), 
                dist, (L2_DISTANCE),
                algo, (KMEANS_MINIBATCH))
        {

            // K = N^(2/3)（取整，至少 8）
            int k = std::max(8, (int)std::round(std::pow((double)0.1 * n, 2.0 / 3.0)));
            int dtype = 0;
            if ((size_t)n * dim > POOL_SIZE) {
                fprintf(stderr, "Error: n=%d * dim=%d = %zu exceeds pool_size=%zu\n", 
                        n, dim, (size_t)n * dim, POOL_SIZE);
                std::abort();
            }
            KMeansCase cfg;
            cfg.n = n;
            cfg.dim = dim;
            cfg.k = k;
            cfg.iters = 5;
            cfg.minibatch_iters = 20;
            cfg.seed = 1234;
            cfg.dist = dist;

            if (!QUIET) {
                COUT_ENDL("========================================");
                COUT_VAL("KMeans Test: n=", cfg.n,
                        " k=", cfg.k,
                        " dim=", cfg.dim,
                        " iters=", cfg.iters,
                        " dist=", (cfg.dist == L2_DISTANCE ? "L2" : "COSINE"),
                        " algo=", (algo == KMEANS_LLOYD ? "LLOYD" : "MINIBATCH")
                        );
                COUT_ENDL("========================================");
            }
    
            auto row = metrics.add_row_averaged([&]() -> std::vector<double> {
                // run_case内部会处理归一化（如果需要），不会修改池数据
                auto r = run_case(cfg, h_pool_data, algo);
                return {
                    r[0],                         // pass_rate
                    (double)cfg.n,
                    (double)cfg.k,
                    (double)cfg.dim,
                    (double)cfg.iters,
                    (double)cfg.dist,
                    (double)algo,                 // algorithm version
                    r[1],                         // gpu_ms
                    r[2],                         // cpu_ms
                    r[3],                         // speedup
                    r[4],                         // rel_obj_err
                    r[5],                         // centroid_rmse
                    r[6],                         // gpu_obj
                    r[7],                         // cpu_obj
                };
            });
            

        }
        // 释放pinned memory池（在当前维度测试完成后）
        cudaFreeHost(h_pool_data);
    }

    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;

    metrics.print_table();
    metrics.export_csv("kmeans_cluster_metrics.csv");
    COUT_ENDL("KMeans tests completed successfully!");
    return 0;
}