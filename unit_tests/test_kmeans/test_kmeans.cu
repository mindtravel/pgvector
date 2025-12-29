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

    // 根据算法版本选择不同的GPU实现
    MEASURE_MS_AND_SAVE("gpu_kmeans耗时:", gpu_ms,
        if (algo == KMEANS_LLOYD) {
            // 全量 Lloyd 算法：处理所有数据点
            gpu_kmeans_lloyd(cfg, h_data, d_assign, d_centroids, &gpu_obj);
        } else {
            // Minibatch 算法：每次迭代只使用一个minibatch
            gpu_kmeans_minibatch(cfg, h_data, d_assign, d_centroids, &gpu_obj);
        }
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );

    // Copy GPU results back
    std::vector<int> h_assign_gpu(n);
    std::vector<float> h_centroids_gpu((size_t)k * dim);
    cudaMemcpy(h_assign_gpu.data(), d_assign, sizeof(int) * (size_t)n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids_gpu.data(), d_centroids, sizeof(float) * (size_t)k * dim, cudaMemcpyDeviceToHost);

    // ====== 测试向量重排功能（使用新的permutation API） ======
    double reorder_ms = 0.0;
    bool reorder_success = true;
    
    // Copy assign back to host
    std::vector<int> h_assign_host(n);
    cudaMemcpy(h_assign_host.data(), d_assign, sizeof(int) * (size_t)n, cudaMemcpyDeviceToHost);
    
    // Build permutation on GPU (only processes assign, not vectors)
    std::vector<int> h_perm(n);
    ClusterInfo h_cluster_info;
    const int BATCH_SIZE = 1 << 20;  // 1M per batch
    
    MEASURE_MS_AND_SAVE("gpu_build_permutation耗时:", reorder_ms,
        gpu_build_permutation_by_cluster(cfg, h_assign_host.data(), h_perm.data(), 
                                        &h_cluster_info, 0, BATCH_SIZE, 0);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );
    
    // 验证cluster信息
    bool ok_info = true;
    if (h_cluster_info.offsets && h_cluster_info.counts && h_cluster_info.k > 0) {
        long long total = 0;
        for (int c = 0; c < h_cluster_info.k; ++c) {
            int cnt = h_cluster_info.counts[c];
            if (cnt < 0 || cnt > n) {
                ok_info = false;
                if (!QUIET) {
                    fprintf(stderr, "Error: Invalid count[%d] = %d\n", c, cnt);
                }
            }
            if (h_cluster_info.offsets[c] != total) {
                ok_info = false;
                if (!QUIET) {
                    fprintf(stderr, "Error: Non-contiguous offsets: offset[%d]=%lld, expected=%lld\n",
                           c, h_cluster_info.offsets[c], total);
                }
            }
            total += (long long)cnt;
        }
        if (total != (long long)n) {
            ok_info = false;
            if (!QUIET) {
                fprintf(stderr, "Error: Total count mismatch: %lld != %d\n", total, n);
            }
        }
    } else {
        ok_info = false;
        if (!QUIET) {
            fprintf(stderr, "Error: Cluster info is null or invalid\n");
        }
    }
    
    // 验证perm是否按cluster非递减排列
    bool ok_perm = true;
    if (ok_info) {
        int prev = -1;
        for (int p = 0; p < n; ++p) {
            int i = h_perm[p];
            if (i < 0 || i >= n) {
                ok_perm = false;
                if (!QUIET && p < 10) {
                    fprintf(stderr, "Error: Invalid perm[%d] = %d\n", p, i);
                }
                continue;
            }
            int c = h_assign_host[i];
            if (c < 0 || c >= k) {
                ok_perm = false;
                if (!QUIET && p < 10) {
                    fprintf(stderr, "Error: Invalid assign[perm[%d]] = assign[%d] = %d\n", p, i, c);
                }
                continue;
            }
            if (c < prev) {
                ok_perm = false;
                if (!QUIET && p < 10) {
                    fprintf(stderr, "Error: Perm not monotonic: perm[%d]=%d (c=%d) < prev c=%d\n", p, i, c, prev);
                }
                break;
            }
            prev = c;
        }
    }
    
    // 构建重排后的向量（多线程CPU实现，使用pageable memory）
    // 必须保证重排后的向量在内存中完整出现
    bool output_built = false;
    float* h_data_reordered = nullptr;
    
    if (ok_info && ok_perm) {
        size_t reordered_size = (size_t)n * (size_t)dim;
        size_t reordered_bytes = reordered_size * sizeof(float);
        
        if (!QUIET) {
            fprintf(stderr, "[test] Allocating reordered vectors (pageable): %.2f GB\n",
                   reordered_bytes / 1024.0 / 1024.0 / 1024.0);
        }
        
        // 使用pageable memory（aligned_alloc），不需要pinned memory
        h_data_reordered = (float*)std::aligned_alloc(64, reordered_bytes);
        if (!h_data_reordered) {
            reorder_success = false;
            if (!QUIET) {
                fprintf(stderr, "[test] ERROR: Failed to alloc h_data_reordered (%.2f GB). "
                       "Cannot guarantee reordered vectors in memory.\n",
                       reordered_bytes / 1024.0 / 1024.0 / 1024.0);
            }
        } else {
            if (!QUIET) {
                fprintf(stderr, "[test] Building reordered vectors on CPU (MT)...\n");
            }
            
            // 多线程重排：out[p] = in[perm[p]]
            const int num_threads = std::max(1u, std::thread::hardware_concurrency());
            const int chunk = (n + num_threads - 1) / num_threads;
            std::vector<std::thread> th;
            th.reserve(num_threads);
            
            for (int t = 0; t < num_threads; ++t) {
                th.emplace_back([=]() {
                    int p0 = t * chunk;
                    int p1 = std::min(n, p0 + chunk);
                    for (int p = p0; p < p1; ++p) {
                        int i = h_perm[p];
                        if (i < 0 || i >= n) continue;
                        const float* src = h_data + (size_t)i * dim;
                        float* dst = h_data_reordered + (size_t)p * dim;
                        std::memcpy(dst, src, sizeof(float) * (size_t)dim);
                    }
                });
            }
            for (auto& x : th) x.join();
            
            output_built = true;
            
            if (!QUIET) {
                fprintf(stderr, "[test] Reordered vectors built successfully. "
                       "Sample: out[0..3] first dim0: %f %f %f %f\n",
                       h_data_reordered[0], h_data_reordered[(size_t)dim], 
                       h_data_reordered[(size_t)2 * dim], h_data_reordered[(size_t)3 * dim]);
            }
        }
    }
    
    // 确保重排后的向量在内存中完整出现（如果permutation成功）
    if (ok_info && ok_perm && !output_built) {
        reorder_success = false;
        if (!QUIET) {
            fprintf(stderr, "[test] ERROR: Reordered vectors not built despite successful permutation.\n");
        }
    } else {
        reorder_success = ok_info && ok_perm && output_built;
    }
    
    // 清理重排后的向量（如果需要保留，可以注释掉这部分）
    if (h_data_reordered) {
        std::free(h_data_reordered);
        h_data_reordered = nullptr;
    }
    
    // 释放cluster信息
    free_cluster_info(&h_cluster_info, false);
    
    if (!QUIET) {
        if (reorder_success) {
            COUT_ENDL("Permutation test passed! (reorder_ms=", reorder_ms, "ms)");
        } else {
            COUT_ENDL("Permutation test FAILED!");
        }
    }

    cudaFree(d_assign);
    cudaFree(d_init_centroids);
    cudaFree(d_centroids);
    
    // 释放临时分配的 pinned memory
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
        const size_t MAX_N = 1000000000;
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
        
        PARAM_3D(n, (20000, 1000000, 10000000, 100000000, 1000000000), 
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