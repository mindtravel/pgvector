#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <limits>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../../cuda/integrate_screen/integrate_screen.cuh"
#include "../../cuda/utils.cuh"
#include "../../cuda/dataset/dataset.cuh"
#include "../common/test_utils.cuh"
#include "../common/cpu_search.h"

struct BenchmarkCase {
    int n_query;
    int vector_dim;
    int n_total_clusters;
    int n_total_vectors;
    int n_probes;   // 粗筛 n_probes
    int k;   // 精筛topk
};

struct BalancedMapping {
    std::vector<int> cluster2block_offset;
    std::vector<int> cluster2block_ids;
    std::vector<int> cluster2block_local_offsets;
    std::vector<int> block_vector_counts;
    int block_count = 0;
};

static BalancedMapping build_balanced_mapping(const ClusterDataset& dataset,
                                              int chunk_size) {
    BalancedMapping mapping;
    int n_total_clusters = dataset.cluster_info.k;
    mapping.cluster2block_offset.resize(n_total_clusters + 1, 0);

    int global_block_id = 0;
    for (int cid = 0; cid < n_total_clusters; ++cid) {
        mapping.cluster2block_offset[cid] =
            static_cast<int>(mapping.cluster2block_ids.size());
        int remaining = dataset.cluster_info.counts[cid];
        int local_offset = 0;
        while (remaining > 0) {
            int take = std::min(chunk_size, remaining);
            mapping.cluster2block_ids.push_back(global_block_id++);
            mapping.cluster2block_local_offsets.push_back(local_offset);
            mapping.block_vector_counts.push_back(take);
            remaining -= take;
            local_offset += take;
        }
    }
    mapping.cluster2block_offset[n_total_clusters] =
        static_cast<int>(mapping.cluster2block_ids.size());
    mapping.block_count = global_block_id;
    return mapping;
}

/**
 * 运行单个测试用例，返回性能指标
 * @param config 测试配置
 * @param dataset 预生成的数据集（包含聚类数据和centroids）
 * @param query_batch 预生成的query批次
 * @return {pass_rate, gpu_ms, cpu_ms, speedup}
 */
static std::vector<double> run_case(const BenchmarkCase& config,
                                     ClusterDataset* dataset,
                                     float** query_batch,
                                     int distance_mode) {
    int** cpu_idx = (int**)malloc_vector_list(config.n_query, config.k, sizeof(int));
    float** cpu_dist = (float**)malloc_vector_list(config.n_query, config.k, sizeof(float));

    int** gpu_idx = (int**)malloc_vector_list(config.n_query, config.k, sizeof(int));
    float** gpu_dist = (float**)malloc_vector_list(config.n_query, config.k, sizeof(float));

    // ClusterDataset 已经包含统一格式的数据，直接使用
    float* reordered_data = dataset->reordered_data;
    ClusterInfo& cluster_info = dataset->cluster_info;
    
    // 展平 query_batch 和 cluster_center_data
    float* query_batch_flat = (float*)malloc(config.n_query * config.vector_dim * sizeof(float));
    for (int i = 0; i < config.n_query; i++) {
        memcpy(query_batch_flat + i * config.vector_dim, query_batch[i], config.vector_dim * sizeof(float));
    }
    
    // 直接使用 dataset->centroids（已经是连续存储的）
    float* centroids_flat = dataset->centroids;
    
    double cpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("cpu耗时:", cpu_ms,
        cpu_coarse_fine_search(config.n_query, config.vector_dim, config.n_total_clusters,
                              config.n_probes, config.k,
                              query_batch_flat, reordered_data, centroids_flat,
                              cluster_info, (DistanceType)distance_mode,
                              cpu_idx, cpu_dist);
    );
    
    // 清理临时内存（reordered_data、cluster_info 和 centroids 由 ClusterDataset 管理，不需要释放）
    free(query_batch_flat);

    // 分配 device 内存并复制数据
    float* d_query_batch = nullptr;
    int* d_cluster_size = nullptr;
    float* d_cluster_vectors = nullptr;
    float* d_cluster_centers = nullptr;
    float* d_topk_dist = nullptr;
    int* d_topk_index = nullptr;
    
    // 1. 复制 query_batch（连续存储）
    cudaMalloc(&d_query_batch, config.n_query * config.vector_dim * sizeof(float));
    // query_batch 是 float**，需要展平为连续数组
    float* query_batch_flat = (float*)malloc(config.n_query * config.vector_dim * sizeof(float));
    for (int i = 0; i < config.n_query; i++) {
        memcpy(query_batch_flat + i * config.vector_dim, query_batch[i], config.vector_dim * sizeof(float));
    }
    cudaMemcpy(d_query_batch, query_batch_flat, config.n_query * config.vector_dim * sizeof(float), cudaMemcpyHostToDevice);
    free(query_batch_flat);
    
    // 2. 复制 cluster_sizes（从 ClusterInfo 获取）
    cudaMalloc(&d_cluster_size, config.n_total_clusters * sizeof(int));
    cudaMemcpy(d_cluster_size, cluster_info.counts, config.n_total_clusters * sizeof(int), cudaMemcpyHostToDevice);
    
    // 3. 复制所有 cluster 向量（直接使用连续存储的数据）
    cudaMalloc(&d_cluster_vectors, config.n_total_vectors * config.vector_dim * sizeof(float));
    cudaMemcpy(d_cluster_vectors, dataset->reordered_data, 
               config.n_total_vectors * config.vector_dim * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // 4. 复制 cluster centers（直接使用 dataset->centroids，已经是连续存储的）
    cudaMalloc(&d_cluster_centers, config.n_total_clusters * config.vector_dim * sizeof(float));
    cudaMemcpy(d_cluster_centers, dataset->centroids, 
               config.n_total_clusters * config.vector_dim * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // 5. 分配输出 device 内存
    cudaMalloc(&d_topk_dist, config.n_query * config.k * sizeof(float));
    cudaMalloc(&d_topk_index, config.n_query * config.k * sizeof(int));
    
    // 6. 在GPU上生成初始索引（cluster索引）：[0, 1, 2, ..., n_total_clusters-1]
    int* d_initial_indices = nullptr;
    cudaMalloc(&d_initial_indices, config.n_query * config.n_total_clusters * sizeof(int));
    CHECK_CUDA_ERRORS;
    
    // 使用kernel在GPU上生成顺序索引
    dim3 block(256);
    dim3 grid((config.n_query * config.n_total_clusters + block.x - 1) / block.x);
    generate_sequential_indices_kernel<<<grid, block>>>(
        d_initial_indices, config.n_query, config.n_total_clusters);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;

    double gpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("gpu耗时:", gpu_ms,
        batch_search_pipeline(
            d_query_batch,
            d_cluster_size,
            d_cluster_vectors,
            d_cluster_centers,
            d_initial_indices,  // 传入初始索引
            d_topk_dist,
            d_topk_index,
            config.n_query,
            config.vector_dim,
            config.n_total_clusters,
            config.n_total_vectors,
            config.n_probes,  // 粗筛选择的cluster数
            config.k,     // k: 最终输出的topk数量
            distance_mode //COSINE_DISTANCE or L2_DISTANCE, 默认COSINE_DISTANCE为cos
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );
    
    // 7. 将结果从 device 复制回 host
    cudaMemcpy(gpu_dist[0], d_topk_dist, config.n_query * config.k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_idx[0], d_topk_index, config.n_query * config.k * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;
    
    // 8. 释放 device 内存
    cudaFree(d_query_batch);
    cudaFree(d_cluster_size);
    cudaFree(d_cluster_vectors);
    cudaFree(d_cluster_centers);
    cudaFree(d_initial_indices);
    cudaFree(d_topk_dist);
    cudaFree(d_topk_index);
    
    bool pass = (distance_mode == L2_DISTANCE) ? compare_set_2D_relative<float>(cpu_dist, gpu_dist, config.n_query, config.k, 1e-4f) :
                       compare_set_2D<float>(cpu_dist, gpu_dist, config.n_query, config.k, 1e-5f);
    
    double pass_rate = pass ? 1.0 : 0.0;

    // 清理内存
    free_vector_list((void**)cpu_idx);
    free_vector_list((void**)cpu_dist);

    free_vector_list((void**)gpu_idx);
    free_vector_list((void**)gpu_dist);

    double speedup = (gpu_ms > 1e-6) ? (cpu_ms / gpu_ms) : 0.0;

    return {
        pass_rate,
        gpu_ms,
        cpu_ms,
        speedup,
    };
}

int main(int argc, char** argv) {
    MetricsCollector metrics;
    metrics.set_columns("pass_rate", "n_query", "n_total_clusters", "vector_dim", 
                        "k", "n_probes", "n_total_vectors", "distance_mode", "gpu_ms", "cpu_ms", 
                        "speedup");
    metrics.set_num_repeats(1);
    
    // 修复：确保 n_total_vectors >= n_total_clusters，这样每个cluster至少有一个向量
    // BenchmarkCase config = {1, 128, 10, 10000, 5, 10};  // n_query=1, dim=128, n_clusters=10, n_vectors=100, n_probes=5, k=10
    // run_case(config); // warmup
    // COUT_ENDL("=========warmup done=========");

    // 缓存的数据集和query
    ClusterDataset cached_dataset = {};
    float** cached_query_batch = nullptr;
    
    // 缓存的关键参数
    int cached_n_total_vectors = -1;
    int cached_vector_dim = -1;
    int cached_n_query = -1;
    
    // PARAM_3D(n_total_vectors, (10000, 50000, 100000, 200000, 500000, 1000000),
    //          n_query, (100, 200, 512, 1000, 2000),
    //          vector_dim, (128, 256))
    // PARAM_3D(n_total_vectors, (10000, 50000, 100000, 200000, 500000, 1000000),
    //         n_query, (100, 200, 512, 1000),
    //         vector_dim, (128, 256))
    PARAM_3D(n_total_vectors, (10000, 1000000),
             n_probes, (1, 2, 5, 10, 20),
             distance_mode, (COSINE_DISTANCE, L2_DISTANCE))  // COSINE_DISTANCE: cosine, L2_DISTANCE: l2
            //  distance_mode, (1))  // 0: cosine, 1: l2
    {
        int n_total_clusters = std::max(10, static_cast<int>(std::sqrt(n_total_vectors)));         
        int n_query = 10;
        int k = 100;
        int vector_dim = 128;
    // 使用和 pgvector 相同的参数进行测试
    // {
    //     int n_query = 4;
    //     int vector_dim = 96;
    //     int n_total_clusters = 50;
    //     int n_total_vectors = 1000;
    //     int n_probes = 8;
    //     int k = 3;
         
        BenchmarkCase config = {n_query, vector_dim, n_total_clusters, n_total_vectors, n_probes, k};
        
        bool need_regenerate_query = (cached_n_query != n_query || 
                                   cached_vector_dim != vector_dim);
        
        bool need_regenerate_dataset = (cached_n_total_vectors != n_total_vectors ||
                                       cached_vector_dim != vector_dim);
        
        if (need_regenerate_dataset) {
            if (cached_dataset.is_valid()) {
                cached_dataset.release();
            }
            cached_dataset.init_with_kmeans(
                n_total_vectors,
                vector_dim,
                n_total_clusters,
                nullptr,  // h_objective
                20,  // kmeans_iters
                false,  // use_minibatch
                COSINE_DISTANCE  // distance_mode
            );
            cached_n_total_vectors = n_total_vectors;
            cached_vector_dim = vector_dim;
        }
        

        if (need_regenerate_query) {
            if (cached_query_batch != nullptr) {
                free_vector_list((void**)cached_query_batch);
            }
            cached_query_batch = generate_vector_list(n_query, vector_dim);
            cached_n_query = n_query;
            cached_vector_dim = vector_dim;
        }
        
        if (!QUIET) {
            COUT_ENDL("========================================");
            COUT_VAL("Testing: n_total_vectors=", n_total_vectors,
                     " n_query=", n_query, 
                     " n_total_clusters=", n_total_clusters,
                     " vector_dim=", vector_dim,
                     " k=", k,
                     " n_probes=", n_probes,
                     " distance_mode=", distance_mode
                    );
            if (need_regenerate_dataset) {
                COUT_ENDL("[INFO] Regenerated dataset");
            }
            if (need_regenerate_query) {
                COUT_ENDL("[INFO] Regenerated query batch");
            }
            COUT_ENDL("========================================");
        }
        
        auto result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto metrics_result = run_case(config, &cached_dataset, cached_query_batch, distance_mode);
            
            std::vector<double> return_vec = {
                metrics_result[0],  // pass_rate
                static_cast<double>(n_query),
                static_cast<double>(n_total_clusters),
                static_cast<double>(vector_dim),
                static_cast<double>(k),
                static_cast<double>(n_probes),
                static_cast<double>(n_total_vectors),
                static_cast<double>(distance_mode),
                metrics_result[1],  // gpu_ms
                metrics_result[2],  // cpu_ms
                metrics_result[3]  // speedup
            };
            return return_vec;
        });
    }
    
    // 清理缓存的数据
    if (cached_dataset.is_valid()) {
        cached_dataset.release();
    }
    if (cached_query_batch != nullptr) {
        free_vector_list((void**)cached_query_batch);
    }
    
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    
    // 打印统计表格
    metrics.print_table();
    
    // 导出 CSV
    metrics.export_csv("integrated_coarse_fine_metrics.csv");
    
    COUT_ENDL("All tests completed successfully!");
    return 0;
}
