#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../../cuda/ivf_search/ivf_search.cuh"
#include "../common/test_utils.cuh"
#include "../cpu_utils/cpu_utils.h"

struct BenchmarkCase {
    int n_query;
    int n_clusters;
    int vector_dim;
    int nprobes;   // 粗筛 nprobes
    int topk;   // 精筛topk
};

struct ClusterDataset {
    float*** cluster_ptrs;
    int* cluster_sizes;
    int n_clusters;
};

struct BalancedMapping {
    std::vector<int> cluster2block_offset;
    std::vector<int> cluster2block_ids;
    std::vector<int> cluster2block_local_offsets;
    std::vector<int> block_vector_counts;
    int block_count = 0;
};

static ClusterDataset prepare_cluster_dataset(const BenchmarkCase& config,
                                              int total_vectors,
                                              std::mt19937& rng) {
    // 根据总向量数和cluster数量，分配每个cluster的向量数
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);

    ClusterDataset dataset;
    dataset.cluster_ptrs = (float***)malloc(config.n_clusters * sizeof(float**));
    dataset.cluster_sizes = (int*)malloc(config.n_clusters * sizeof(int));

    // 平均分配向量到各个cluster，剩余部分随机分配
    int base_vec_per_cluster = total_vectors / config.n_clusters;
    int remainder = total_vectors % config.n_clusters;
    
    // 先分配基础数量
    for (int cid = 0; cid < config.n_clusters; ++cid) {
        dataset.cluster_sizes[cid] = base_vec_per_cluster;
    }
    
    // 随机分配剩余向量
    std::uniform_int_distribution<int> remainder_dist(0, config.n_clusters - 1);
    for (int i = 0; i < remainder; ++i) {
        dataset.cluster_sizes[remainder_dist(rng)]++;
    }

    // 为每个cluster分配内存并生成数据
    for (int cid = 0; cid < config.n_clusters; ++cid) {
        int vec_count = dataset.cluster_sizes[cid];
        dataset.cluster_ptrs[cid] = const_cast<float**>(generate_vector_list(vec_count, config.vector_dim));
    }
    return dataset;
}

static void release_cluster_dataset(ClusterDataset& dataset) {
    for (int cid = 0; cid < dataset.n_clusters; ++cid) {
        free_vector_list((void**)dataset.cluster_ptrs[cid]);
    }
    free(dataset.cluster_ptrs);
    free(dataset.cluster_sizes);
}

static BalancedMapping build_balanced_mapping(const ClusterDataset& dataset,
                                              int chunk_size) {
    BalancedMapping mapping;
    int n_clusters = dataset.n_clusters;
    mapping.cluster2block_offset.resize(n_clusters + 1, 0);

    int global_block_id = 0;
    for (int cid = 0; cid < n_clusters; ++cid) {
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
    mapping.cluster2block_offset[n_clusters] =
        static_cast<int>(mapping.cluster2block_ids.size());
    mapping.block_count = global_block_id;
    return mapping;
}

struct QueryBatch {
    float** ptrs = nullptr;
};




static void cpu_coarse_fine_search(const BenchmarkCase& config,
                                   float** query_batch,
                                   const ClusterDataset& dataset,
                                   float** centers,
                                   int n_cluster_per_query,
                                   int nprobes,
                                   int** out_index,
                                   float** out_dist
                                ) {
    const int n_query = config.n_query;
    const int n_dim = config.vector_dim;
    const int n_clusters = config.n_clusters;

    struct Pair { float dist; int idx; };  

    const unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());
    auto worker = [&](int start, int end) {
        std::vector<Pair> tmp(n_clusters);
        std::vector<Pair> fine_buffer;
        for (int qi = start; qi < end; ++qi) {
            const float* query = query_batch[qi];

            // coarse: compare with cluster centers
            for (int cid = 0; cid < n_clusters; ++cid) {
                tmp[cid] = {cosine_distance(query, centers[cid], n_dim), cid};
            }
            std::partial_sort(tmp.begin(), tmp.begin() + n_cluster_per_query, tmp.end(),
                              [](const Pair& a, const Pair& b) { return a.dist < b.dist; });

            // fine: iterate vectors inside selected clusters
            fine_buffer.clear();
            
            for (int k = 0; k < n_cluster_per_query; ++k) {
                int cid = tmp[k].idx;  // 从partial_sort后的tmp数组获取cluster ID
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
            if (fine_buffer.size() > static_cast<size_t>(nprobes)) {
                // 使用 nth_element 选择前 nprobes 个最小的元素
                std::nth_element(fine_buffer.begin(), fine_buffer.begin() + nprobes, fine_buffer.end(),
                                 [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
                fine_buffer.resize(nprobes);
                // nth_element 只保证第 n 个元素在正确位置，需要排序
                std::sort(fine_buffer.begin(), fine_buffer.end(),
                          [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            } else {
                // 候选数不足 nprobes，直接排序
                std::sort(fine_buffer.begin(), fine_buffer.end(),
                          [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            }

            for (int t = 0; t < nprobes; ++t) {
                if (t < static_cast<int>(fine_buffer.size())) {
                    out_index[qi][t] = fine_buffer[t].idx;
                    out_dist[qi][t] = fine_buffer[t].dist;
                } else {
                    out_index[qi][t] = -1;
                    out_dist[qi][t] = 0.0f;
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
 * @param total_vectors 总向量数
 * @return {pass_rate_unbalanced, pass_rate_balanced, gpu_ms_unbalanced, gpu_ms_balanced, cpu_ms, speedup_unbalanced, speedup_balanced, total_vectors}
 */
static std::vector<double> run_case(const BenchmarkCase& config, int total_vectors) {
    std::mt19937 rng(1234);

    auto dataset = prepare_cluster_dataset(config, total_vectors, rng);
    
    float** query_batch = const_cast<float**>(generate_vector_list(config.n_query, config.vector_dim));
    auto balanced = build_balanced_mapping(dataset, 512);

    float*** cluster_data = dataset.cluster_ptrs;
    int* cluster_sizes = dataset.cluster_sizes;
    float** cluster_center_data = const_cast<float**>(generate_vector_list(config.n_clusters, config.vector_dim));

    // CPU 参考实现：使用 C 数组
    // 注意：输出大小应该是 topk（最终输出的topk数量），不是 nprobes（粗筛的cluster数）
    int** cpu_idx = (int**)malloc_vector_list(config.n_query, config.topk, sizeof(int));
    float** cpu_dist = (float**)malloc_vector_list(config.n_query, config.topk, sizeof(float));


    double cpu_ms = 0.0;
    MEASURE_MS_AND_SAVE("cpu耗时:", cpu_ms,
        cpu_coarse_fine_search(config, query_batch, dataset, cluster_center_data,
                               config.nprobes,  // n_cluster_per_query: 粗筛选择的cluster数
                               config.topk,     // nprobes: 最终输出的topk数量
                               cpu_idx,
                               cpu_dist
        );
    );
    
    auto run_pipeline = [&](bool use_balance, const char* tag, int** topk_index,
                            float** topk_dist) -> double {
        int* n_isnull = static_cast<int*>(malloc(config.n_query * sizeof(int)));

        double gpu_ms = 0.0;
        MEASURE_MS_AND_SAVE("gpu耗时:", gpu_ms,
            ivf_search_pipeline(
                query_batch,
                cluster_sizes,
                cluster_data,
                cluster_center_data,

                topk_dist,
                topk_index,
                n_isnull,

                config.n_query,
                config.vector_dim,
                config.n_clusters,
                config.nprobes,  // n_cluster_per_query: 粗筛选择的cluster数
                config.topk,     // k: 最终输出的topk数量

                use_balance,
                use_balance ? balanced.cluster2block_offset.data() : nullptr,
                use_balance ? balanced.cluster2block_ids.data() : nullptr,
                use_balance ? balanced.cluster2block_local_offsets.data() : nullptr,
                use_balance ? balanced.block_vector_counts.data() : nullptr,
                use_balance ? balanced.block_count : 0
            );
            cudaDeviceSynchronize();
            CHECK_CUDA_ERRORS;
        );

        if (!QUIET) {
            COUT_VAL("[", tag, "] n_query=", config.n_query,
                     " clusters=", config.n_clusters,
                     " dim=", config.vector_dim,
                     " elapsed=", gpu_ms, "ms");
            COUT_VAL("  sample query0 isnull=", n_isnull[0],
                     " top_index[0]=", topk_index[0][0],
                     " top_dist[0]=", topk_dist[0][0]);
        }

        free(n_isnull);
        
        return gpu_ms;
    };

    // GPU 输出大小也应该是 topk
    int** gpu_idx_unbalanced = (int**)malloc_vector_list(config.n_query, config.topk, sizeof(int));
    float** gpu_dist_unbalanced = (float**)malloc_vector_list(config.n_query, config.topk, sizeof(float));
    // int** gpu_idx_balanced = (int**)malloc_vector_list(config.n_query, config.topk, sizeof(int));
    // float** gpu_dist_balanced = (float**)malloc_vector_list(config.n_query, config.topk, sizeof(float));

    double gpu_ms_unbalanced = run_pipeline(false, "unbalanced", gpu_idx_unbalanced, gpu_dist_unbalanced);
    // double gpu_ms_balanced = run_pipeline(true, "balanced", gpu_idx_balanced, gpu_dist_balanced);

    // 比较时使用 topk 作为输出大小
    // 添加调试输出
    if (config.n_query <= 4 && config.topk <= 2) {
        COUT_ENDL("=== DEBUG: Comparing CPU vs GPU results ===");
        for (int qi = 0; qi < config.n_query; ++qi) {
            COUT_VAL("Query ", qi, ":");
            COUT_VAL("  CPU:   ");
            for (int j = 0; j < config.topk; ++j) {
                COUT_VAL("(idx=", cpu_idx[qi][j], " dist=", cpu_dist[qi][j], ") ");
            }
            COUT_ENDL();
            COUT_VAL("  GPU(unbalanced): ");
            for (int j = 0; j < config.topk; ++j) {
                COUT_VAL("(idx=", gpu_idx_unbalanced[qi][j], " dist=", gpu_dist_unbalanced[qi][j], ") ");
            }
            COUT_ENDL();
            // COUT_VAL("  GPU(balanced):   ");
            // for (int j = 0; j < config.topk; ++j) {
            //     COUT_VAL("(idx=", gpu_idx_balanced[qi][j], " dist=", gpu_dist_balanced[qi][j], ") ");
            // }
            // COUT_ENDL();
        }
    }
    
    // 只比较距离数组
    bool pass_unbalanced = compare_set_2D<float>(cpu_dist, gpu_dist_unbalanced, config.n_query, config.topk, 1e-3f);
    // bool pass_balanced = compare_set_2D<float>(cpu_dist, gpu_dist_balanced, config.n_query, config.topk, 1e-3f);
    
    double pass_rate_unbalanced = pass_unbalanced ? 1.0 : 0.0;
    // double pass_rate_balanced = pass_balanced ? 1.0 : 0.0;
    
    // 清理内存
    free_vector_list((void**)cpu_idx);
    free_vector_list((void**)cpu_dist);

    free_vector_list((void**)gpu_idx_unbalanced);
    free_vector_list((void**)gpu_dist_unbalanced);
    // free_vector_list((void**)gpu_idx_balanced);
    // free_vector_list((void**)gpu_dist_balanced);

    double speedup_unbalanced = (gpu_ms_unbalanced > 1e-6) ? (cpu_ms / gpu_ms_unbalanced) : 0.0;
    // double speedup_balanced = (gpu_ms_balanced > 1e-6) ? (cpu_ms / gpu_ms_balanced) : 0.0;

    free_vector_list((void**)query_batch);
    release_cluster_dataset(dataset);

    return {
        pass_rate_unbalanced,
        // pass_rate_balanced,
        gpu_ms_unbalanced,
        // gpu_ms_balanced,
        cpu_ms,
        speedup_unbalanced,
        // speedup_balanced,
    };
}

int main() {
    MetricsCollector metrics;
    // metrics.set_columns("pass_rate_unbal", "pass_rate_bal", "n_query", "n_clusters", "vector_dim", 
    //                     "topk", "nprobes", "total_vectors", "gpu_ms_unbal", "gpu_ms_bal", 
    //                     "cpu_ms", "speedup_unbal", "speedup_bal");
    metrics.set_columns("pass_rate_unbal", "n_query", "n_clusters", "vector_dim", 
                        "topk", "nprobes", "total_vectors", "gpu_ms_unbal", "cpu_ms", 
                        "speedup_unbal");
    metrics.set_num_repeats(1);
    
    BenchmarkCase config = {100, 10, 128, 5, 10};
    
    config = {100, 10, 128, 5, 10};
    run_case(config, 10000); // warmup
    COUT_ENDL("=========warmup done=========");

    // PARAM_3D(total_vectors, (10000, 50000, 100000, 200000, 500000, 1000000),
    //          n_query, (100, 200, 512, 1000, 2000),
    //          vector_dim, (128, 256))
    // PARAM_3D(total_vectors, (10000, 50000, 100000, 200000, 500000, 1000000),
    //         n_query, (100, 200, 512, 1000),
    //         vector_dim, (128, 256))
    //  PARAM_3D(total_vectors, (10000, 1000000),
    //          nprobes, (1, 2, 5, 10, 20),
    //          vector_dim, (128))
    // {
    //     int n_clusters = std::max(10, static_cast<int>(std::sqrt(total_vectors)));         
    //     int n_query = 10000;
    //     int topk = 100;
    PARAM_3D(total_vectors, (10000),
    nprobes, (5),
    vector_dim, (128))
    {
        int n_clusters = std::max(10, static_cast<int>(std::sqrt(total_vectors)));         
        int n_query = 4;
        int topk = 2;
         
        BenchmarkCase config = {n_query, n_clusters, vector_dim, nprobes, topk};
        
        if (!QUIET) {
            COUT_ENDL("========================================");
            COUT_VAL("Testing: total_vectors=", total_vectors,
                     " n_query=", n_query, 
                     " n_clusters=", n_clusters,
                     " vector_dim=", vector_dim,
                     " topk=", topk,
                     " nprobes=", nprobes);
            COUT_ENDL("========================================");
        }
        
        try {
            auto result = metrics.add_row_averaged([&]() -> std::vector<double> {
                auto metrics_result = run_case(config, total_vectors);
                // 返回：n_query, n_clusters, vector_dim, topk, nprobes, total_vectors, pass_unbal, pass_bal, 
                //       gpu_ms_unbal, gpu_ms_bal, cpu_ms, speedup_unbal, speedup_bal
                // return {
                //     metrics_result[0],  // pass_rate_unbalanced
                //     // metrics_result[1],  // pass_rate_balanced
                //     static_cast<double>(n_query),
                //     static_cast<double>(n_clusters),
                //     static_cast<double>(vector_dim),
                //     static_cast<double>(topk),
                //     static_cast<double>(nprobes),
                //     static_cast<double>(total_vectors),

                //     metrics_result[2],  // gpu_ms_unbalanced
                //     // metrics_result[3],  // gpu_ms_balanced
                //     metrics_result[4],  // cpu_ms
                //     metrics_result[5],  // speedup_unbalanced
                //     // metrics_result[6]   // speedup_balanced
                // };
                return {
                    metrics_result[0],  // pass_rate_unbalanced
                    static_cast<double>(n_query),
                    static_cast<double>(n_clusters),
                    static_cast<double>(vector_dim),
                    static_cast<double>(topk),
                    static_cast<double>(nprobes),
                    static_cast<double>(total_vectors),
                    metrics_result[1],  // gpu_ms_unbalanced
                    metrics_result[2],  // cpu_ms
                    metrics_result[3]  // speedup_unbalanced
                };
            });
            
            // if (!QUIET) {
            //     COUT_VAL("Result: total_vectors=", static_cast<int>(result[7]),
            //              " pass_rate_unbal=", std::fixed, std::setprecision(4), result[0], 
            //              " pass_rate_bal=", std::fixed, std::setprecision(4), result[1],
            //              " speedup_unbal=", std::fixed, std::setprecision(2), result[11]
            //             //  " speedup_bal=", std::fixed, std::setprecision(2), result[12]
            //             );
            // }
        } catch (const std::exception& e) {
            COUT_VAL("[ERROR] Test failed: ", e.what());
            return 1;
        }
    }
    
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    
    // 打印统计表格
    metrics.print_table();
    
    // 导出 CSV
    metrics.export_csv("ivf_search_coarse_fine_balance_metrics.csv");
    
    COUT_ENDL("All tests completed successfully!");
    return 0;
}
