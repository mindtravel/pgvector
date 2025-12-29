#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
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

// ============================================================
// Utilities (CPU)
// ============================================================

static void cpu_kmeans_lloyd(
    const KMeansCase& cfg,
    const float* data,            // [n, dim]
    int* out_assign,              // [n]
    float* centroids,         // [k, dim]
    float* out_objective          // sum of min dist^2
) {
    const int n = cfg.n, d = cfg.dim, k = cfg.k;
    
    // 获取线程数
    const int num_threads = std::max(1, (int)std::thread::hardware_concurrency());
    

    for (int it = 0; it < cfg.iters; ++it) {
        // 每个线程的局部累加缓冲区
        std::vector<std::vector<float>> thread_centroids(num_threads);
        std::vector<std::vector<int>> thread_counts(num_threads);
        std::vector<double> thread_obj(num_threads, 0.0);
        for (int t = 0; t < num_threads; ++t) {
            thread_centroids[t].resize(k * d, 0.0f);
            thread_counts[t].resize(k, 0);
        }
        
        // 并行 assign 和 accumulate
        std::vector<std::thread> workers;
        const int chunk_size = (n + num_threads - 1) / num_threads;
        
        for (int tid = 0; tid < num_threads; ++tid) {
            workers.emplace_back([&, tid]() {
                const int start = tid * chunk_size;
                const int end = std::min(start + chunk_size, n);
                
                double local_obj = 0.0;
                float* local_centroids = thread_centroids[tid].data();
                int* local_counts = thread_counts[tid].data();
                
                for (int i = start; i < end; ++i) {
                    const float* x = data + (size_t)i * d;
                    int best = 0;
                    float best_dist = l2_distance_squared(x, centroids, d);
                    
                    // 找最近的 centroid
                    for (int c = 1; c < k; ++c) {
                        float dist = l2_distance_squared(x, centroids + (size_t)c * d, d);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best = c;
                        }
                    }
                    
                    out_assign[i] = best;
                    local_obj += (double)best_dist;
                    
                    // 累加到线程局部缓冲区
                    float* acc = local_centroids + (size_t)best * d;
                    for (int j = 0; j < d; ++j) {
                        acc[j] += x[j];
                    }
                    local_counts[best] += 1;
                }
                
                thread_obj[tid] = local_obj;
            });
        }
        
        // 等待所有线程完成
        for (auto& t : workers) {
            t.join();
        }
        
        // 合并所有线程的累加结果
        std::vector<float> next_centroids(k * d, 0.0f);
        std::vector<int> counts(k, 0);
        double obj_sum = 0.0;
        
        for (int t = 0; t < num_threads; ++t) {
            obj_sum += thread_obj[t];
            for (int c = 0; c < k; ++c) {
                counts[c] += thread_counts[t][c];
                float* dst = next_centroids.data() + (size_t)c * d;
                const float* src = thread_centroids[t].data() + (size_t)c * d;
                for (int j = 0; j < d; ++j) {
                    dst[j] += src[j];
                }
            }
        }

        // 并行更新 centroids
        std::vector<std::thread> update_workers;
        const int centroid_chunk = (k + num_threads - 1) / num_threads;
        
        for (int tid = 0; tid < num_threads; ++tid) {
            update_workers.emplace_back([&, tid]() {
                const int start = tid * centroid_chunk;
                const int end = std::min(start + centroid_chunk, k);
                
                for (int c = start; c < end; ++c) {
                    float* cc = centroids + (size_t)c * d;
                    if (counts[c] > 0) {
                        float inv = 1.0f / (float)counts[c];
                        for (int j = 0; j < d; ++j) {
                            cc[j] = next_centroids[(size_t)c * d + j] * inv;
                        }
                    }
                    // else: keep previous centroid
                }
            });
        }
        
        for (auto& t : update_workers) {
            t.join();
        }

        if (out_objective) *out_objective = (float)obj_sum;
    }
}



// ============================================================
// Test Case Runner
// ============================================================
static std::vector<double> run_case(const KMeansCase& cfg) {
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;

    // host data (pinned memory for faster H2D transfer)
    float* h_data = nullptr;
    cudaMallocHost(&h_data, sizeof(float) * (size_t)n * dim);
    {
        std::mt19937 rng(cfg.seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < (size_t)n * dim; ++i) h_data[i] = dist(rng);
    }

    // cosine -> normalize vectors
    if (cfg.dist == COSINE_DISTANCE) {
        for (int i = 0; i < n; ++i) {
            l2_normalize_inplace(h_data + (size_t)i * dim, dim);
        }
    }

    // 初始化聚类中心：使用统一的初始化函数确保CPU和GPU使用相同的起始聚类中心
    // 该函数使用 cfg.seed 确保确定性初始化（使用 pinned memory）
    float* h_init_centroids = nullptr;
    cudaMallocHost(&h_init_centroids, sizeof(float) * (size_t)k * dim);
    init_centroids_by_sampling(cfg, h_data, h_init_centroids);
    if (cfg.dist == COSINE_DISTANCE) {
        for (int c = 0; c < k; ++c) l2_normalize_inplace(h_init_centroids + (size_t)c * dim, dim);
    }

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
    int* d_assign = nullptr;
    float* d_init_centroids = nullptr;
    float* d_centroids = nullptr;

    cudaMalloc(&d_assign, sizeof(int) * (size_t)n);
    cudaMalloc(&d_init_centroids, sizeof(float) * (size_t)k * dim);
    cudaMalloc(&d_centroids, sizeof(float) * (size_t)k * dim);
    // 将CPU初始化的聚类中心复制到GPU，确保CPU和GPU使用相同的初始化
    // 使用异步拷贝可以利用 pinned memory 的优势
    cudaMemcpyAsync(d_centroids, h_init_centroids, sizeof(float) * (size_t)k * dim, 
                   cudaMemcpyHostToDevice);

    float gpu_obj = 0.0f;
    double gpu_ms = 0.0;

    // 现在函数接受主机端数据指针（pinned memory），内部使用双缓冲分片上传
    MEASURE_MS_AND_SAVE("gpu_kmeans耗时:", gpu_ms,
        gpu_kmeans_lloyd(cfg, h_data, d_assign, d_centroids, &gpu_obj);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );

    // Copy GPU results back
    std::vector<int> h_assign_gpu(n);
    std::vector<float> h_centroids_gpu((size_t)k * dim);
    cudaMemcpy(h_assign_gpu.data(), d_assign, sizeof(int) * (size_t)n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids_gpu.data(), d_centroids, sizeof(float) * (size_t)k * dim, cudaMemcpyDeviceToHost);

    cudaFree(d_assign);
    cudaFree(d_init_centroids);
    cudaFree(d_centroids);
    
    // 释放 pinned memory
    cudaFreeHost(h_data);
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
    double cen_tol = 1e-4;

    // bool pass = (centroid_rmse < cen_tol);
    bool pass = rel_obj_err < obj_tol;

    double pass_rate = pass ? 1.0 : 0.0;
    double speedup = (gpu_ms > 1e-6) ? (cpu_ms / gpu_ms) : 0.0;

    if (!QUIET) {
        COUT_ENDL("----- KMeans Verify -----");
        COUT_VAL("cpu_obj=", cpu_obj, " gpu_obj=", gpu_obj, " rel_obj_err=", rel_obj_err);
        COUT_VAL("centroid_rmse=", centroid_rmse, " pass=", (pass ? 1 : 0));
        COUT_ENDL("-------------------------");
    }

    return {pass_rate, gpu_ms, cpu_ms, speedup, (double)rel_obj_err, (double)centroid_rmse, (double)gpu_obj, (double)cpu_obj};
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
    metrics.set_columns("pass_rate", "n", "k", "dim", "iters", "dist",
                        "gpu_ms", "cpu_ms", "speedup", "rel_obj_err", "centroid_rmse",
                        "gpu_obj", "cpu_obj");
    metrics.set_num_repeats(1);

    // PARAM_3D(n, (10000, 20000),
    //  dim, (4,10,96,128,200),
    PARAM_3D(n, (20000, 1000000, 10000000),
             dim, (96),
             dist, (L2_DISTANCE))
    {
        int iters = 5;
        // K = N^(2/3)（取整，至少 8）
        int k = std::max(8, (int)std::round(std::pow((double)0.1 * n, 2.0 / 3.0)));
        int dtype = 0;
        // 为了单测别太慢，可以给 K 做个上限（你要测大K就放开）
        // k = std::min(k, 4096);

        KMeansCase cfg;
        cfg.n = n;
        cfg.dim = dim;
        cfg.k = k;
        cfg.iters = iters;
        cfg.seed = 1234;
        cfg.dist = dist;

        if (!QUIET) {
            COUT_ENDL("========================================");
            COUT_VAL("KMeans Test: n=", cfg.n,
                    " k=", cfg.k,
                    " dim=", cfg.dim,
                    " iters=", cfg.iters,
                    " dist=", (cfg.dist == L2_DISTANCE ? "L2" : "COSINE")
                    );
            COUT_ENDL("========================================");
        }

        auto row = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto r = run_case(cfg);
            return {
                r[0],                         // pass_rate
                (double)cfg.n,
                (double)cfg.k,
                (double)cfg.dim,
                (double)cfg.iters,
                (double)cfg.dist,
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

    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;

    metrics.print_table();
    metrics.export_csv("kmeans_cluster_metrics.csv");
    COUT_ENDL("KMeans tests completed successfully!");
    return 0;
}