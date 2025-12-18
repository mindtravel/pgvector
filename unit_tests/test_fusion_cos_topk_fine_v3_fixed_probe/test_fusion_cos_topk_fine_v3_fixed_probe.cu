#include <stdlib.h>
#include <limits>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <ctime>
#include <cfloat>
#include <cmath>
#include <random>
#include <thread>

#include "../../cuda/fusion_cos_topk/fusion_cos_topk.cuh"
#include "../../cuda/pch.h"
#include "../common/test_utils.cuh"
#include "../common/params_macros.cuh"
#include "../common/output_macros.cuh"

//#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>

#define EPSILON 1e-2f

/**
 * CPU版本余弦距离top-k计算（用于验证）- 多线程版本
 * 
 * 对于 fixed_probe 版本，需要：
 * 1. 对每个 query，遍历所有 probe
 * 2. 对每个 probe，遍历该 probe 的所有向量
 * 3. 计算余弦距离并选择 top-k（使用 partial_sort）
 * 4. 合并所有 probe 的结果，选择全局 top-k（使用 partial_sort）
 */
void cpu_cos_distance_topk_fine_v3_fixed_probe(
    float** query_vectors,
    float** cluster_vectors,
    int* probe_vector_offset,
    int* probe_vector_count,
    int* probe_queries,
    int* probe_query_offsets,
    int* probe_query_probe_indices,
    int* query_clusters,  // 新增：每个query的probe对应的cluster [n_query][n_probes]
    float* query_norm,
    float* cluster_vector_norm,
    int** topk_index,
    float** topk_dist,
    float** candidate_dist,  // 新增：候选距离 [n_query][n_probes * k]
    int** candidate_index,    // 新增：候选索引 [n_query][n_probes * k]
    int n_query,
    int n_total_clusters,
    int n_probes,  // 新增：每个query的probe数量
    int n_dim,
    int k
) {
    // 获取可用线程数
    unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
    num_threads = std::min(num_threads, static_cast<unsigned int>(n_query));
    
    // 每个线程处理一个或多个query
    auto process_query_range = [&](int start_q, int end_q) {
        for (int q = start_q; q < end_q; q++) {
            // 为每个 probe 存储候选结果
            std::vector<std::vector<std::pair<float, int>>> probe_candidates(n_probes);
            std::vector<std::pair<float, int>> all_candidates;
            
            // 直接遍历每个query的每个probe，找到对应的cluster
            for (int p = 0; p < n_probes; p++) {
                int cluster_id = query_clusters[q * n_probes + p];
                if (cluster_id < 0 || cluster_id >= n_total_clusters) continue;
                
                // 遍历该 cluster 的所有向量
                int vec_start = probe_vector_offset[cluster_id];
                int vec_end = vec_start + probe_vector_count[cluster_id];
                
                float* query = query_vectors[q];
                float query_n = query_norm[q];
                
                for (int v = vec_start; v < vec_end; v++) {
                    float dot_product = 0.0f;
                    const float* vec_ptr = cluster_vectors[v];
                    
                    for (int d = 0; d < n_dim; d++) {
                        dot_product += query[d] * vec_ptr[d];
                    }
                    
                    float vec_n = cluster_vector_norm[v];
                    if (vec_n < 1e-6f || query_n < 1e-6f) continue;
                    
                    // query_n 和 vec_n 已经是开过根号的L2范数，不需要再开根号
                    float cos_similarity = dot_product / (query_n * vec_n);
                    float cos_distance = 1.0f - cos_similarity;
                    
                    probe_candidates[p].emplace_back(cos_distance, v);
                    all_candidates.emplace_back(cos_distance, v);
                }
            }
            
            // 为每个 probe 选择 top-k 候选（使用 partial_sort）
            if (candidate_dist != nullptr && candidate_index != nullptr) {
                for (int p = 0; p < n_probes; p++) {
                    int probe_size = static_cast<int>(probe_candidates[p].size());
                    if (probe_size > 0) {
                        int topk_count = std::min(k, probe_size);
                        // 使用 partial_sort 只排序前 k 个元素
                        std::partial_sort(
                            probe_candidates[p].begin(),
                            probe_candidates[p].begin() + topk_count,
                            probe_candidates[p].end()
                        );
                        for (int i = 0; i < topk_count; i++) {
                            candidate_dist[q][p * k + i] = probe_candidates[p][i].first;
                            candidate_index[q][p * k + i] = probe_candidates[p][i].second;
                        }
                    }
                    // 填充剩余位置为无效值
                    for (int i = probe_size; i < k; i++) {
                        candidate_dist[q][p * k + i] = FLT_MAX;
                        candidate_index[q][p * k + i] = -1;
                    }
                }
            }
            
            // 使用 partial_sort 选择全局 top-k
            int all_size = static_cast<int>(all_candidates.size());
            if (all_size > 0) {
                int topk_count = std::min(k, all_size);
                // 使用 partial_sort 只排序前 k 个元素
                std::partial_sort(
                    all_candidates.begin(),
                    all_candidates.begin() + topk_count,
                    all_candidates.end()
                );
                for (int i = 0; i < topk_count; i++) {
                    topk_dist[q][i] = all_candidates[i].first;
                    topk_index[q][i] = all_candidates[i].second;
                }
            }
            // 填充剩余位置为无效值
            for (int i = all_size; i < k; i++) {
                topk_dist[q][i] = FLT_MAX;
                topk_index[q][i] = -1;
            }
        }
    };
    
    // 分配任务给各个线程
    std::vector<std::thread> threads;
    int queries_per_thread = (n_query + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; t++) {
        int start_q = t * queries_per_thread;
        int end_q = std::min(start_q + queries_per_thread, n_query);
        if (start_q < n_query) {
            threads.emplace_back(process_query_range, start_q, end_q);
        }
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
}

/**
 * 测试单个参数组合
 */
std::vector<double> test_single_config(
    int n_query, 
    int n_probes, 
    int n_total_clusters, 
    int n_dim, 
    int k, 
    int vectors_per_cluster,
    float** h_cluster_vectors = nullptr,
    float* h_cluster_vector_norm = nullptr
) {
    // 1. 生成测试数据
    float** h_query_vectors = generate_vector_list(n_query, n_dim);
    int n_total_vectors = n_total_clusters * vectors_per_cluster;



    float* h_query_norm = (float*)malloc(n_query * sizeof(float));
    // 计算query向量的L2范数（开过根号）
    compute_l2_norms_batch(h_query_vectors, h_query_norm, n_query, n_dim);
    
    // 跟踪哪些内存是在函数内部分配的，需要在清理时释放
    bool need_free_cluster_vectors = false;
    bool need_free_cluster_vector_norm = false;
    
    if(h_cluster_vectors == nullptr) {
        h_cluster_vectors = generate_vector_list(n_total_vectors, n_dim);
        need_free_cluster_vectors = true;
    }
    
    if(h_cluster_vector_norm == nullptr) {
        h_cluster_vector_norm = (float*)malloc(n_total_vectors * sizeof(float));
        need_free_cluster_vector_norm = true;
    }
    
    // 如果向量数据存在但L2范数未计算，需要计算
    if(h_cluster_vectors != nullptr && h_cluster_vector_norm != nullptr) {
        compute_l2_norms_batch(h_cluster_vectors, h_cluster_vector_norm, n_total_vectors, n_dim);
    }
    
    // 3. 为每个 query 随机生成 n_probes 个 query-cluster 对
    // 使用随机数生成器确保每个 query 都有不同的 cluster 组合
    int* query_clusters = (int*)malloc(n_query * n_probes * sizeof(int));  // 每个 query 对应的 cluster 列表
    
    for (int q = 0; q < n_query; q++) {
        // 为每个 query 随机选择 n_probes 个不同的 cluster
        std::vector<int> available_clusters;
        for (int c = 0; c < n_total_clusters; c++) {
            available_clusters.push_back(c);
        }
        
        // 随机打乱并选择前 n_probes 个
        std::random_shuffle(available_clusters.begin(), available_clusters.end());
        for (int i = 0; i < n_probes && i < (int)available_clusters.size(); i++) {
            query_clusters[q * n_probes + i] = available_clusters[i];
        }
    }
    
    // 4. 构建 probe-query 映射（CSR 格式）
    // probe 在这里指的是 cluster，每个 cluster 对应一个 probe
    // 需要统计每个 cluster 被哪些 query 使用
    // 使用纯C数组实现，不使用C++容器
    
    // 第一步：统计每个 cluster 有多少个 query 使用它
    int* cluster_query_count = (int*)calloc(n_total_clusters, sizeof(int));  // 初始化为0
    for (int q = 0; q < n_query; q++) {
        for (int probe_idx = 0; probe_idx < n_probes; probe_idx++) {
            int cluster_id = query_clusters[q * n_probes + probe_idx];
            if (cluster_id >= 0 && cluster_id < n_total_clusters) {
                cluster_query_count[cluster_id]++;
            }
        }
    }
    
    // 第二步：构建 CSR 格式的 offsets 数组
    int total_query_cluster_pairs = n_query * n_probes;
    int* probe_query_offsets = (int*)malloc((n_total_clusters + 1) * sizeof(int));
    probe_query_offsets[0] = 0;
    for (int c = 0; c < n_total_clusters; c++) {
        probe_query_offsets[c + 1] = probe_query_offsets[c] + cluster_query_count[c];
    }
    
    // 第三步：分配输出数组
    int* probe_queries = (int*)malloc(total_query_cluster_pairs * sizeof(int));
    int* probe_query_probe_indices = (int*)malloc(total_query_cluster_pairs * sizeof(int));
    
    // 第四步：使用临时数组记录每个 cluster 当前写入位置
    int* cluster_write_pos = (int*)malloc(n_total_clusters * sizeof(int));
    for (int c = 0; c < n_total_clusters; c++) {
        cluster_write_pos[c] = probe_query_offsets[c];
    }
    
    // 第五步：遍历所有 query-cluster 对，填充数据
    for (int q = 0; q < n_query; q++) {
        for (int probe_idx = 0; probe_idx < n_probes; probe_idx++) {
            int cluster_id = query_clusters[q * n_probes + probe_idx];
            if (cluster_id >= 0 && cluster_id < n_total_clusters) {
                int write_pos = cluster_write_pos[cluster_id];
                probe_queries[write_pos] = q;  // query_id
                probe_query_probe_indices[write_pos] = probe_idx;  // probe_index_in_query
                cluster_write_pos[cluster_id]++;
            }
        }
    }
    
    // 清理临时数组
    free(cluster_query_count);
    free(cluster_write_pos);
    
    // 6. 构建 probe（cluster）向量映射
    int* probe_vector_offset = (int*)malloc(n_total_clusters * sizeof(int));
    int* probe_vector_count = (int*)malloc(n_total_clusters * sizeof(int));
    for (int c = 0; c < n_total_clusters; c++) {
        probe_vector_offset[c] = c * vectors_per_cluster;
        probe_vector_count[c] = vectors_per_cluster;
    }
    
    // 7. CPU 参考实现
    float** h_topk_dist_cpu = (float**)malloc_vector_list(n_query, k, sizeof(float));
    int** h_topk_index_cpu = (int**)malloc_vector_list(n_query, k, sizeof(int));
    
    // CPU 候选结果缓冲区
    float** candidate_dist_cpu = (float**)malloc_vector_list(n_query, n_probes * k, sizeof(float));
    int** candidate_index_cpu = (int**)malloc_vector_list(n_query, n_probes * k, sizeof(int));
    
    double cpu_duration_ms = 0;
    MEASURE_MS_AND_SAVE("CPU fine search", cpu_duration_ms,
        cpu_cos_distance_topk_fine_v3_fixed_probe(
            h_query_vectors,
            h_cluster_vectors,
            probe_vector_offset,
            probe_vector_count,
            probe_queries,
            probe_query_offsets,
            probe_query_probe_indices,
            query_clusters,  // 传递query_clusters映射
            h_query_norm,
            h_cluster_vector_norm,
            h_topk_index_cpu,
            h_topk_dist_cpu,
            candidate_dist_cpu,
            candidate_index_cpu,
            n_query,
            n_total_clusters,
            n_probes,
            n_dim,
            k
        );
    );
    //nvtxRangePushA("GPU");
    // 6. GPU 实现
    float* d_query_group = nullptr;
    float* d_cluster_vector = nullptr;
    int* d_probe_vector_offset = nullptr;
    int* d_probe_vector_count = nullptr;
    int* d_probe_queries = nullptr;
    int* d_probe_query_offsets = nullptr;
    int* d_probe_query_probe_indices = nullptr;
    float* d_query_norm = nullptr;
    float* d_cluster_vector_norm = nullptr;
    int* d_topk_index = nullptr;
    float* d_topk_dist = nullptr;
    
    // 分配设备内存
    cudaMalloc(&d_query_group, n_query * n_dim * sizeof(float));
    cudaMalloc(&d_cluster_vector, n_total_vectors * n_dim * sizeof(float));
    cudaMalloc(&d_probe_vector_offset, n_total_clusters * sizeof(int));
    cudaMalloc(&d_probe_vector_count, n_total_clusters * sizeof(int));
    int total_pairs = probe_query_offsets[n_total_clusters];  // 实际使用的 query-cluster 对数量
    cudaMalloc(&d_probe_queries, total_pairs * sizeof(int));
    cudaMalloc(&d_probe_query_offsets, (n_total_clusters + 1) * sizeof(int));
    cudaMalloc(&d_probe_query_probe_indices, total_pairs * sizeof(int));
    cudaMalloc(&d_query_norm, n_query * sizeof(float));
    cudaMalloc(&d_cluster_vector_norm, n_total_vectors * sizeof(float));
    cudaMalloc(&d_topk_index, n_query * k * sizeof(int));
    cudaMalloc(&d_topk_dist, n_query * k * sizeof(float));
    CHECK_CUDA_ERRORS;
    
    // 复制数据到设备
    cudaMemcpy(d_query_group, h_query_vectors[0], n_query * n_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vector, h_cluster_vectors[0], n_total_vectors * n_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_vector_offset, probe_vector_offset, n_total_clusters * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_vector_count, probe_vector_count, n_total_clusters * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_queries, probe_queries, total_pairs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_query_offsets, probe_query_offsets, (n_total_clusters + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_query_probe_indices, probe_query_probe_indices, total_pairs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_norm, h_query_norm, n_query * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vector_norm, h_cluster_vector_norm, n_total_vectors * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;
    
    // 准备候选结果缓冲区（用于调试输出）
    float** candidate_dist = (float**)malloc_vector_list(n_query, n_probes * k, sizeof(float));
    int** candidate_index = (int**)malloc_vector_list(n_query, n_probes * k, sizeof(int));
    
    // GPU kernel 执行
    double gpu_duration_ms = 0;

    MEASURE_MS_AND_SAVE("GPU fine search", gpu_duration_ms,
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
            d_topk_index,
            d_topk_dist,
            candidate_dist,
            candidate_index,
            n_query,
            n_total_clusters, 
            n_probes,
            n_dim,
            k
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );

    // 复制结果回主机
    float** h_topk_dist_gpu = (float**)malloc_vector_list(n_query, k, sizeof(float));
    int** h_topk_index_gpu = (int**)malloc_vector_list(n_query, k, sizeof(int));
    
    cudaMemcpy(h_topk_dist_gpu[0], d_topk_dist, n_query * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_topk_index_gpu[0], d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;
    //nvtxRangePop();
    // 7. 验证结果（使用 test_utils.cuh 的 compare_set_2D）
    bool pass = true;
    pass &= compare_set_2D(h_topk_dist_gpu, h_topk_dist_cpu, n_query, k, EPSILON);
    // 注意：索引比较可能因为距离相同而顺序不同，这里只比较距离
    // 如果需要比较索引，可以使用 compare_set_2D(h_topk_index_gpu, h_topk_index_cpu, n_query, k)
    
    // 验证候选结果
    bool candidate_pass = true;
    if (candidate_dist != nullptr && candidate_index != nullptr && 
        candidate_dist_cpu != nullptr && candidate_index_cpu != nullptr) {
        // 先进行详细比较，找出不一致的位置
        int mismatch_count = 0;
        const int max_mismatches_to_print = 20;
        for (int q = 0; q < n_query; q++) {
            for (int p = 0; p < n_probes; p++) {
                for (int ki = 0; ki < k; ki++) {
                    int idx = p * k + ki;
                    float gpu_dist = candidate_dist[q][idx];
                    float cpu_dist = candidate_dist_cpu[q][idx];
                    int gpu_idx = candidate_index[q][idx];
                    int cpu_idx = candidate_index_cpu[q][idx];
                    
                    if (fabs(gpu_dist - cpu_dist) > EPSILON) {
                        if (mismatch_count < max_mismatches_to_print) {
                            printf("[MISMATCH] Query %d, Probe %d, k_idx %d: GPU(dist=%.6f, idx=%d) vs CPU(dist=%.6f, idx=%d), diff=%.6f\n",
                                   q, p, ki, gpu_dist, gpu_idx, cpu_dist, cpu_idx, fabs(gpu_dist - cpu_dist));
                        }
                        mismatch_count++;
                    }
                }
            }
        }
        if (mismatch_count > 0) {
            printf("[ERROR] Found %d candidate mismatches (showing first %d)\n", 
                   mismatch_count, max_mismatches_to_print);
            candidate_pass = false;
        } else {
            candidate_pass = true;
        }
        
        // 也使用 compare_set_2D 进行集合比较（忽略顺序）
        // candidate_pass = compare_set_2D(candidate_dist, candidate_dist_cpu, n_query, n_probes * k, EPSILON);
    }
    
    // 8. 清理
    cudaFree(d_query_group);
    cudaFree(d_cluster_vector);
    cudaFree(d_probe_vector_offset);
    cudaFree(d_probe_vector_count);
    cudaFree(d_probe_queries);
    cudaFree(d_probe_query_offsets);
    cudaFree(d_probe_query_probe_indices);
    cudaFree(d_query_norm);
    cudaFree(d_cluster_vector_norm);
    cudaFree(d_topk_index);
    cudaFree(d_topk_dist);
    
    free_vector_list((void**)h_query_vectors);
    if(need_free_cluster_vectors) {
        free_vector_list((void**)h_cluster_vectors);
    }
    free_vector_list((void**)h_topk_dist_cpu);
    free_vector_list((void**)h_topk_index_cpu);
    free_vector_list((void**)h_topk_dist_gpu);
    free_vector_list((void**)h_topk_index_gpu);
    free_vector_list((void**)candidate_dist);
    free_vector_list((void**)candidate_index);
    free_vector_list((void**)candidate_dist_cpu);
    free_vector_list((void**)candidate_index_cpu);
    free(h_query_norm);
    if(need_free_cluster_vector_norm) {
        free(h_cluster_vector_norm);
    }
    free(query_clusters);
    free(probe_query_offsets);
    free(probe_queries);
    free(probe_query_probe_indices);
    free(probe_vector_offset);
    free(probe_vector_count);
    
    // 9. 返回结果
    double pass_rate = pass ? 1.0 : 0.0;
    double candidate_pass_rate = candidate_pass ? 1.0 : 0.0;
    double speedup = cpu_duration_ms > 0 ? cpu_duration_ms / gpu_duration_ms : 0.0;
    double memory_mb = (double)(n_query * n_dim + n_total_vectors * n_dim) * sizeof(float) / (double)(1024 * 1024);
    
    return {pass_rate, candidate_pass_rate, (double)n_query, (double)n_probes, (double)n_total_clusters, (double)n_dim, (double)k, 
            gpu_duration_ms, cpu_duration_ms, speedup, memory_mb};
}

int main(int argc, char** argv) {
    srand(time(0));
    
    bool all_pass = true;
    MetricsCollector metrics;
    metrics.set_columns("pass rate", "candidate_pass", "n_query", "n_probes", "n_total_clusters", "n_dim", "k", 
                        "avg_gpu_ms", "avg_cpu_ms", "avg_speedup", "memory_mb");
    
    // Warmup
    test_single_config(8, 4, 16, 128, 10, 32);
    
    COUT_ENDL("测试算法: cuda_cos_topk_warpsort_fine_v3_fixed_probe");
    metrics.set_num_repeats(1);

    int n_dim = 128;
    int n_total_clusters = 1024;  // 总聚类数
    int vectors_per_cluster = 1024;  // 每个 cluster 的向量数

    int n_total_vectors = n_total_clusters * vectors_per_cluster;
    float** h_cluster_vectors = generate_vector_list(n_total_vectors, n_dim);
    float* h_cluster_vector_norm = (float*)malloc(n_total_vectors * sizeof(float));
    compute_l2_norms_batch(h_cluster_vectors, h_cluster_vector_norm, n_total_vectors, n_dim);

    // 参数扫描
    PARAM_3D(n_query, (10000),
             n_probes, (1, 5, 10, 20, 40),
             k, (100))
    // PARAM_3D(n_query, (8, 32, 128),
    //     n_probes, (1, 5, 10, 20, 40),
    //     k, (10, 20))
    {
        COUT_ENDL("n_query: ", n_query, ", n_probes: ", n_probes, ", k: ", k);
        auto avg_result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto result = test_single_config(
                n_query, n_probes, n_total_clusters, n_dim, k, vectors_per_cluster,
                h_cluster_vectors, h_cluster_vector_norm
            );
            all_pass &= (result[0] == 1.0);  // 检查 pass 字段（topk结果）
            all_pass &= (result[1] == 1.0);  // 检查 candidate_pass 字段（候选结果）
            return result;
        });
    }
    
    metrics.print_table();
    
    COUT_ENDL(all_pass ? "✅ All tests passed!" : "❌ Some tests failed!");
    return all_pass ? 0 : 1;
}