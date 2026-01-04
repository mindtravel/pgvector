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
// Mismatch Analysis Functions
// ============================================================

/**
 * 通过向量索引找到它所属的聚类
 * @param vector_idx 向量在重排数据中的全局索引
 * @param cluster_info 聚类信息
 * @return 聚类ID，如果未找到返回-1
 */
int find_cluster_by_vector_idx(int vector_idx, const ClusterInfo& cluster_info) {
    for (int cid = 0; cid < cluster_info.k; ++cid) {
        long long offset = cluster_info.offsets[cid];
        int count = cluster_info.counts[cid];
        if (vector_idx >= offset && vector_idx < offset + count) {
            return cid;
        }
    }
    return -1;
}

/**
 * 分析不匹配的元素，输出详细信息和CPU/GPU各自选出的前10个最近的聚类
 * @param mismatches 不匹配信息列表
 * @param cpu_idx CPU版本的索引数组
 * @param cpu_dist CPU版本的距离数组
 * @param gpu_idx GPU版本的索引数组
 * @param gpu_dist GPU版本的距离数组
 * @param cpu_coarse_idx CPU版本的粗筛结果索引
 * @param cpu_coarse_dist CPU版本的粗筛结果距离
 * @param gpu_coarse_idx GPU版本的粗筛结果索引
 * @param gpu_coarse_dist GPU版本的粗筛结果距离
 * @param n_probes 粗筛选择的聚类数
 * @param topk 精筛返回的topk数量
 * @param dataset 数据集（包含聚类信息）
 * @param query_batch 查询向量
 * @param dist_type 距离类型
 */
void analyze_mismatches(
    const std::vector<MismatchInfo>& mismatches,
    int** cpu_idx,
    float** cpu_dist,
    int** gpu_idx,
    float** gpu_dist,
    int** cpu_coarse_idx,
    float** cpu_coarse_dist,
    int** gpu_coarse_idx,
    float** gpu_coarse_dist,
    int n_probes,
    int topk,
    const ClusterDataset& dataset,
    const float* query_batch,
    DistanceType dist_type) {
    
    if (mismatches.empty()) {
        return;
    }
    
    COUT_ENDL("========================================");
    COUT_ENDL("分析不匹配的元素:");
    COUT_VAL("不匹配数量: ", mismatches.size());
    
    // 统计每个查询的不匹配情况
    std::map<int, std::vector<const MismatchInfo*>> mismatches_by_query;
    for (const auto& mismatch : mismatches) {
        mismatches_by_query[mismatch.query_idx].push_back(&mismatch);
    }
    
    // 分析每个查询的不匹配
    for (const auto& [query_idx, query_mismatches] : mismatches_by_query) {
        COUT_ENDL("----------------------------------------");
        COUT_VAL("查询索引: ", query_idx);
        COUT_VAL("该查询的不匹配数量: ", query_mismatches.size());
        
        // 输出CPU和GPU各自选出的前10个最近的聚类
        COUT_ENDL("----------------------------------------");
        COUT_ENDL("CPU选出的前10个最近的聚类:");
        if (cpu_coarse_idx && cpu_coarse_dist) {
            COUT_TABLE("排名", "聚类ID", "距离");
            for (int i = 0; i < n_probes; ++i) {
                COUT_TABLE(i + 1, cpu_coarse_idx[query_idx][i], cpu_coarse_dist[query_idx][i]);
            }
        } else {
            COUT_ENDL("  (粗筛结果不可用)");
        }
        
        COUT_ENDL("GPU选出的前10个最近的聚类:");
        if (gpu_coarse_idx && gpu_coarse_dist) {
            COUT_TABLE("排名", "聚类ID", "距离");
            for (int i = 0; i < n_probes; ++i) {
                COUT_TABLE(i + 1, gpu_coarse_idx[query_idx][i], gpu_coarse_dist[query_idx][i]);
            }
        } else {
            COUT_ENDL("  (粗筛结果不可用)");
        }

        // 输出详细的不匹配信息
        COUT_ENDL("----------------------------------------");
        COUT_ENDL("CPU和GPU最终结果:");
        COUT_TABLE("位置", "CPU距离", "CPU索引", "CPU聚类", "GPU距离", "GPU索引", "GPU聚类", "距离差异");

        if (cpu_idx && cpu_dist && gpu_idx && gpu_dist) {
            for (int i = 0; i < topk; ++i) {
                int cpu_reordered_pos = cpu_idx[query_idx][i];
                int gpu_reordered_pos = gpu_idx[query_idx][i];
                int cpu_cluster_id = (cpu_reordered_pos >= 0) ? find_cluster_by_vector_idx(cpu_reordered_pos, dataset.cluster_info) : -1;
                int gpu_cluster_id = (gpu_reordered_pos >= 0) ? find_cluster_by_vector_idx(gpu_reordered_pos, dataset.cluster_info) : -1;
                COUT_TABLE(i + 1, cpu_dist[query_idx][i], cpu_reordered_pos, cpu_cluster_id,
                                  gpu_dist[query_idx][i], gpu_reordered_pos, gpu_cluster_id,
                                  std::abs(cpu_dist[query_idx][i] - gpu_dist[query_idx][i]));
            }
        }
    }
    
    COUT_ENDL("========================================");
}

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
    
    // if (!QUIET) {
    //     COUT_VAL("K-means objective: ", kmeans_objective);
    // }
    
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
        cpu_coarse_fine_search(cfg.n_query, cfg.dim, cfg.n_clusters, cfg.n_probes, cfg.topk,
                              h_query_batch, 
                              dataset.reordered_data,    // 使用GPU聚类后重排的数据
                              dataset.centroids,         // 使用GPU聚类的中心
                              dataset.cluster_info,      // 使用GPU聚类的cluster信息
                              cfg.dist,
                              cpu_idx, cpu_dist,
                              cpu_coarse_idx, cpu_coarse_dist);  // 粗筛结果
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
    int** gpu_idx = malloc_vector_list<int>(n_query, topk);
    float** gpu_dist = malloc_vector_list<float>(n_query, topk);
    
    // 分配GPU粗筛结果数组（host端）
    int** h_gpu_coarse_idx = malloc_vector_list<int>(n_query, n_probes);
    float** h_gpu_coarse_dist = malloc_vector_list<float>(n_query, n_probes);
    
    double gpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("GPU搜索耗时:", gpu_ms,
        ivf_search_pipeline(
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
            h_gpu_coarse_dist   // 粗筛结果距离
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );
    
    // Step 5: 复制结果回host（CPU和GPU都返回重排后的位置索引）
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
    
    // Step 9: 如果有不匹配，分析原因
    // if (!pass && !fine_mismatches.empty()) {
    //     analyze_mismatches(fine_mismatches, cpu_idx, cpu_dist, 
    //                       gpu_idx, gpu_dist, 
    //                       cpu_coarse_idx, cpu_coarse_dist,
    //                       h_gpu_coarse_idx, h_gpu_coarse_dist,
    //                       n_probes, topk, dataset, h_query_batch, cfg.dist);
    // }
    
    // 总体通过需要粗筛和精筛都通过
    bool overall_pass = coarse_pass && pass;
    
    double pass_rate = overall_pass ? 1.0 : 0.0;
    double speedup = (gpu_ms > 1e-6) ? (cpu_ms / gpu_ms) : 0.0;
    
    if (!QUIET) {
        COUT_ENDL("----- IVF-Flat Search Verify -----");
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
    metrics.export_csv("ivfflat_search_metrics.csv");
    COUT_ENDL("IVF-Flat Search tests completed successfully!");
    return 0;
}

