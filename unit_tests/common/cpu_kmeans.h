#ifndef CPU_KMEANS_H
#define CPU_KMEANS_H

#include <vector>
#include <thread>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include "../../cuda/kmeans/kmeans.cuh"
#include "cpu_distance.h"

/**
 * CPU K-means参考实现
 * 用于验证GPU实现的正确性
 */

/**
 * CPU K-means Lloyd算法实现（多线程）
 * 
 * @param cfg K-means配置
 * @param data 输入数据 [n, dim]
 * @param out_assign 输出分配结果 [n]
 * @param centroids 输入输出聚类中心 [k, dim]（输入为初始值，输出为最终值）
 * @param out_objective 输出目标函数值（可选）
 */
inline void cpu_kmeans_lloyd(
    const KMeansCase& cfg,
    const float* data,            // [n, dim]
    int* out_assign,              // [n]
    float* centroids,             // [k, dim] (in/out)
    float* out_objective          // sum of min dist^2 (optional)
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

#endif // CPU_KMEANS_H

