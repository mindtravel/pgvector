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
#include "../common/test_utils.cuh"

struct BenchmarkCase {
    int n_query;
    int vector_dim;
    int n_total_clusters;
    int n_total_vectors;
    int n_probes;   // 粗筛 n_probes
    int k;   // 精筛topk
};

struct ClusterDataset {
    float*** cluster_ptrs;
    int* cluster_sizes;
    int* initial_indices;  /* 初始索引（cluster索引）[n_query * n_total_clusters] */
    int n_total_clusters;
};

struct BalancedMapping {
    std::vector<int> cluster2block_offset;
    std::vector<int> cluster2block_ids;
    std::vector<int> cluster2block_local_offsets;
    std::vector<int> block_vector_counts;
    int block_count = 0;
};

static ClusterDataset prepare_cluster_dataset(const BenchmarkCase& config) {
    std::mt19937 rng(1234);

    // 根据总向量数和cluster数量，分配每个cluster的向量数
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);

    ClusterDataset dataset;
    dataset.n_total_clusters = config.n_total_clusters;  // 初始化n_total_clusters
    dataset.cluster_ptrs = (float***)malloc(config.n_total_clusters * sizeof(float**));
    dataset.cluster_sizes = (int*)malloc(config.n_total_clusters * sizeof(int));

    // 平均分配向量到各个cluster，剩余部分随机分配
    int base_vec_per_cluster = config.n_total_vectors / config.n_total_clusters;
    int remainder = config.n_total_vectors % config.n_total_clusters;
    
    // 先分配基础数量
    for (int cid = 0; cid < config.n_total_clusters; ++cid) {
        dataset.cluster_sizes[cid] = base_vec_per_cluster;
    }
    
    // 随机分配剩余向量
    std::uniform_int_distribution<int> remainder_dist(0, config.n_total_clusters - 1);
    for (int i = 0; i < remainder; ++i) {
        dataset.cluster_sizes[remainder_dist(rng)]++;
    }

    // 为每个cluster分配内存并生成数据
    for (int cid = 0; cid < config.n_total_clusters; ++cid) {
        int vec_count = dataset.cluster_sizes[cid];
        dataset.cluster_ptrs[cid] = generate_vector_list(vec_count, config.vector_dim);
    }
    
    // 生成初始索引（cluster索引）：[0, 1, 2, ..., n_total_clusters-1]，每个query使用相同的索引
    dataset.initial_indices = (int*)malloc(config.n_query * config.n_total_clusters * sizeof(int));
    for (int qi = 0; qi < config.n_query; qi++) {
        for (int cid = 0; cid < config.n_total_clusters; cid++) {
            dataset.initial_indices[qi * config.n_total_clusters + cid] = cid;
        }
    }
    
    return dataset;
}

static void release_cluster_dataset(ClusterDataset& dataset) {
    for (int cid = 0; cid < dataset.n_total_clusters; ++cid) {
        free_vector_list((void**)dataset.cluster_ptrs[cid]);
        dataset.cluster_ptrs[cid] = nullptr;
    }
    free(dataset.cluster_ptrs);
    dataset.cluster_ptrs = nullptr;
    free(dataset.cluster_sizes);
    dataset.cluster_sizes = nullptr;
    if (dataset.initial_indices) {
        free(dataset.initial_indices);
        dataset.initial_indices = nullptr;
    }
}

static BalancedMapping build_balanced_mapping(const ClusterDataset& dataset,
                                              int chunk_size) {
    BalancedMapping mapping;
    int n_total_clusters = dataset.n_total_clusters;
    mapping.cluster2block_offset.resize(n_total_clusters + 1, 0);

    int global_block_id = 0;
    for (int cid = 0; cid < n_total_clusters; ++cid) {
        mapping.cluster2block_offset[cid] =
            static_cast<int>(mapping.cluster2block_ids.size());
        int remaining = dataset.cluster_sizes[cid];
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

struct QueryBatch {
    float** ptrs = nullptr;
};
static float cosine_distance(const float* a, const float* b, int dim) {
    float dot = 0.0f;
    float na = 0.0f;
    float nb = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    if (denom < 1e-6f) return 1.0f;
    return 1.0f - dot / denom;
}




static void cpu_coarse_fine_search(const BenchmarkCase& config,
                                   float** query_batch,
                                   const ClusterDataset& dataset,
                                   float** centers,
                                   int* initial_indices,
                                   int** out_index,
                                   float** out_dist
                                ) {
    const int n_query = config.n_query;
    const int n_dim = config.vector_dim;
    const int n_total_clusters = config.n_total_clusters;

    struct Pair { float dist; int idx; };  

    const unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());
    auto worker = [&](int start, int end) {
        std::vector<Pair> tmp(n_total_clusters);
        std::vector<Pair> fine_buffer;
        for (int qi = start; qi < end; ++qi) {
            const float* query = query_batch[qi];

            // coarse: compare with cluster centers (使用传入的初始索引)
            const int* query_initial_indices = initial_indices + qi * n_total_clusters;
            for (int idx = 0; idx < n_total_clusters; ++idx) {
                int cid = query_initial_indices[idx];  // 使用传入的索引
                tmp[idx] = {cosine_distance(query, centers[cid], n_dim), cid};
            }
            // 粗筛选择 n_probes 个 cluster
            std::partial_sort(tmp.begin(), tmp.begin() + config.n_probes, tmp.end(),
                              [](const Pair& a, const Pair& b) { return a.dist < b.dist; });

            // fine: iterate vectors inside selected clusters
            fine_buffer.clear();
            
            // 遍历粗筛选出的 n_probes 个 cluster
            for (int probe_idx = 0; probe_idx < config.n_probes; ++probe_idx) {
                int cid = tmp[probe_idx].idx;  // 从partial_sort后的tmp数组获取cluster ID
                int vec_count = dataset.cluster_sizes[cid];
                const float* base = dataset.cluster_ptrs[cid][0];
                
                // 计算该 cluster 的全局向量偏移
                int cluster_global_offset = 0;
                for (int prev_cid = 0; prev_cid < cid; ++prev_cid) {
                    cluster_global_offset += dataset.cluster_sizes[prev_cid];
                }
                
                for (int vid = 0; vid < vec_count; ++vid) {
                    const float* vec = base + static_cast<size_t>(vid) * n_dim;
                    // 存储全局索引：cluster 的全局偏移 + cluster 内的局部索引
                    int global_idx = cluster_global_offset + vid;
                    fine_buffer.push_back({cosine_distance(query, vec, n_dim), global_idx});
                }
            }
            // 从所有候选向量中选择 top-k
            if (fine_buffer.size() > static_cast<size_t>(config.k)) {
                // 使用 nth_element 选择前 k 个最小的元素
                std::nth_element(fine_buffer.begin(), fine_buffer.begin() + config.k, fine_buffer.end(),
                                 [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
                fine_buffer.resize(config.k);
                // nth_element 只保证第 k 个元素在正确位置，需要排序
                std::sort(fine_buffer.begin(), fine_buffer.end(),
                          [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            } else {
                // 候选数不足 k，直接排序
                std::sort(fine_buffer.begin(), fine_buffer.end(),
                          [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            }

            // 输出 top-k 结果
            if (fine_buffer.empty() && qi == 0) {
                printf("[DEBUG CPU] Query 0: fine_buffer is empty! n_probes=%d\n", config.n_probes);
                for (int probe_idx = 0; probe_idx < config.n_probes; ++probe_idx) {
                    int cid = tmp[probe_idx].idx;
                    int vec_count = dataset.cluster_sizes[cid];
                    printf("[DEBUG CPU] Probe %d: cluster_id=%d, vec_count=%d\n", probe_idx, cid, vec_count);
                }
            }
            for (int t = 0; t < config.k; ++t) {
                if (t < static_cast<int>(fine_buffer.size())) {
                    out_index[qi][t] = fine_buffer[t].idx;
                    out_dist[qi][t] = fine_buffer[t].dist;
                } else {
                    out_index[qi][t] = -1;
                    out_dist[qi][t] = std::numeric_limits<float>::max();  // 使用最大值表示无效值
                }
            }
        }
    };

    std::vector<std::thread> threads;
    int chunk = (n_query + num_threads - 1) / num_threads;
    int start = 0;
    for (unsigned t = 0; t < num_threads && start < n_query; ++t) {
        int end = std::min(n_query, start + chunk);
        threads.emplace_back(worker, start, end);
        start = end;
    }
    for (auto& th : threads) {
        th.join();
    }
}

/**
 * 运行单个测试用例，返回性能指标
 * @param config 测试配置
 * @param dataset 预生成的数据集（如果为nullptr，则内部生成）
 * @param query_batch 预生成的query批次（如果为nullptr，则内部生成）
 * @param cluster_center_data 预生成的cluster中心（如果为nullptr，则内部生成）
 * @return {pass_rate, pass_rate_balanced, gpu_ms, gpu_ms_balanced, cpu_ms, speedup, speedup_balanced, n_total_vectors}
 */
static std::vector<double> run_case(const BenchmarkCase& config,
                                     ClusterDataset* dataset,
                                     float** query_batch,
                                     float** cluster_center_data, 
                                     int mode) {
    float*** cluster_data = dataset->cluster_ptrs;
    int* cluster_sizes = dataset->cluster_sizes;
    int* initial_indices = dataset->initial_indices;  /* 使用数据集中的初始索引 */
    
    int** cpu_idx = (int**)malloc_vector_list(config.n_query, config.k, sizeof(int));
    float** cpu_dist = (float**)malloc_vector_list(config.n_query, config.k, sizeof(float));

    int** gpu_idx = (int**)malloc_vector_list(config.n_query, config.k, sizeof(int));
    float** gpu_dist = (float**)malloc_vector_list(config.n_query, config.k, sizeof(float));

    double cpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("cpu耗时:", cpu_ms,
        cpu_coarse_fine_search(config, query_batch, *dataset,
                               cluster_center_data,
                               initial_indices,  /* 传入初始索引（一维数组） */
                               cpu_idx,
                               cpu_dist
        );
    );

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
    
    // 2. 复制 cluster_sizes
    cudaMalloc(&d_cluster_size, config.n_total_clusters * sizeof(int));
    cudaMemcpy(d_cluster_size, cluster_sizes, config.n_total_clusters * sizeof(int), cudaMemcpyHostToDevice);
    
    // 3. 复制所有 cluster 向量（连续存储）
    cudaMalloc(&d_cluster_vectors, config.n_total_vectors * config.vector_dim * sizeof(float));
    float* cluster_vectors_flat = (float*)malloc(config.n_total_vectors * config.vector_dim * sizeof(float));
    int offset = 0;
    for (int cid = 0; cid < config.n_total_clusters; cid++) {
        int vec_count = cluster_sizes[cid];
        for (int vid = 0; vid < vec_count; vid++) {
            memcpy(cluster_vectors_flat + offset * config.vector_dim, 
                   cluster_data[cid][0] + vid * config.vector_dim, 
                   config.vector_dim * sizeof(float));
            offset++;
        }
    }
    cudaMemcpy(d_cluster_vectors, cluster_vectors_flat, config.n_total_vectors * config.vector_dim * sizeof(float), cudaMemcpyHostToDevice);
    free(cluster_vectors_flat);
    
    // 4. 复制 cluster_center_data（连续存储）
    cudaMalloc(&d_cluster_centers, config.n_total_clusters * config.vector_dim * sizeof(float));
    float* cluster_centers_flat = (float*)malloc(config.n_total_clusters * config.vector_dim * sizeof(float));
    for (int i = 0; i < config.n_total_clusters; i++) {
        memcpy(cluster_centers_flat + i * config.vector_dim, cluster_center_data[i], config.vector_dim * sizeof(float));
    }
    cudaMemcpy(d_cluster_centers, cluster_centers_flat, config.n_total_clusters * config.vector_dim * sizeof(float), cudaMemcpyHostToDevice);
    free(cluster_centers_flat);
    
    // 5. 分配输出 device 内存
    cudaMalloc(&d_topk_dist, config.n_query * config.k * sizeof(float));
    cudaMalloc(&d_topk_index, config.n_query * config.k * sizeof(int));
    
    // 6. 复制初始索引（cluster索引）到GPU（使用数据集中的初始索引）
    int* d_initial_indices = nullptr;
    cudaMalloc(&d_initial_indices, config.n_query * config.n_total_clusters * sizeof(int));
    cudaMemcpy(d_initial_indices, initial_indices, 
               config.n_query * config.n_total_clusters * sizeof(int), 
               cudaMemcpyHostToDevice);
    
    CHECK_CUDA_ERRORS;

    double gpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("gpu耗时:", gpu_ms,
        batch_search_pipeline_choose(
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
            mode //cos or l2 默认0为cos
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
    
    bool pass = compare_set_2D<float>(cpu_dist, gpu_dist, config.n_query, config.k, 1e-4f);
    
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
                        "k", "n_probes", "n_total_vectors", "gpu_ms", "cpu_ms", 
                        "speedup");
    metrics.set_num_repeats(1);
    
    // 修复：确保 n_total_vectors >= n_total_clusters，这样每个cluster至少有一个向量
    // BenchmarkCase config = {1, 128, 10, 10000, 5, 10};  // n_query=1, dim=128, n_clusters=10, n_vectors=100, n_probes=5, k=10
    // run_case(config); // warmup
    // COUT_ENDL("=========warmup done=========");

    // 缓存的数据集和query
    ClusterDataset cached_dataset = {};
    float** cached_query_batch = nullptr;
    float** cached_cluster_center_data = nullptr;
    
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
             vector_dim, (128))
    {
        int n_total_clusters = std::max(10, static_cast<int>(std::sqrt(n_total_vectors)));         
        int n_query = 10000;
        int k = 100;
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
            if (cached_dataset.cluster_ptrs != nullptr) {
                release_cluster_dataset(cached_dataset);
            }
            if (cached_cluster_center_data != nullptr) {
                free_vector_list((void**)cached_cluster_center_data);
                cached_cluster_center_data = nullptr;
            }
            cached_dataset = prepare_cluster_dataset(config);
            cached_cluster_center_data = generate_vector_list(n_total_clusters, vector_dim);
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
                     " n_probes=", n_probes);
            if (need_regenerate_dataset) {
                COUT_ENDL("[INFO] Regenerated dataset");
            }
            if (need_regenerate_query) {
                COUT_ENDL("[INFO] Regenerated query batch");
            }
            COUT_ENDL("========================================");
        }
        
        auto result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto metrics_result = run_case(config, &cached_dataset, cached_query_batch, cached_cluster_center_data,0);
            
            std::vector<double> return_vec = {
                metrics_result[0],  // pass_rate
                static_cast<double>(n_query),
                static_cast<double>(n_total_clusters),
                static_cast<double>(vector_dim),
                static_cast<double>(k),
                static_cast<double>(n_probes),
                static_cast<double>(n_total_vectors),
                metrics_result[1],  // gpu_ms
                metrics_result[2],  // cpu_ms
                metrics_result[3]  // speedup
            };
            return return_vec;
        });
    }
    
    // 清理缓存的数据
    if (cached_dataset.cluster_ptrs != nullptr) {
        release_cluster_dataset(cached_dataset);
    }
    if (cached_query_batch != nullptr) {
        free_vector_list((void**)cached_query_batch);
    }
    if (cached_cluster_center_data != nullptr) {
        free_vector_list((void**)cached_cluster_center_data);
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
