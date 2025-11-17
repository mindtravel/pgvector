#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../../cuda/integrate_screen/integrate_screen.cuh"
#include "../common/test_utils.cuh"

struct BenchmarkCase {
    int n_query;
    int n_clusters;
    int vector_dim;
    int topk;   // 粗筛 cluster 数
    int topn;   // 精筛 top-n
};

struct ClusterDataset {
    std::vector<float*> cluster_ptrs;
    std::vector<int> cluster_sizes;
};

struct BalancedMapping {
    std::vector<int> cluster2block_offset;
    std::vector<int> cluster2block_ids;
    std::vector<int> cluster2block_local_offsets;
    std::vector<int> block_vector_counts;
    int block_count = 0;
};

static ClusterDataset prepare_cluster_dataset(const BenchmarkCase& config,
                                              std::mt19937& rng) {
    std::uniform_int_distribution<int> count_dist(500, 3000);
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);

    ClusterDataset dataset;
    dataset.cluster_ptrs.resize(config.n_clusters);
    dataset.cluster_sizes.resize(config.n_clusters);

    for (int cid = 0; cid < config.n_clusters; ++cid) {
        int vec_count = count_dist(rng);
        dataset.cluster_sizes[cid] = vec_count;

        float* host_ptr = nullptr;
        size_t bytes = static_cast<size_t>(vec_count) * config.vector_dim * sizeof(float);
        cudaMallocHost(reinterpret_cast<void**>(&host_ptr), bytes);
        CHECK_CUDA_ERRORS;

        for (size_t i = 0; i < static_cast<size_t>(vec_count) * config.vector_dim; ++i) {
            host_ptr[i] = value_dist(rng);
        }
        dataset.cluster_ptrs[cid] = host_ptr;
    }
    return dataset;
}

static void release_cluster_dataset(ClusterDataset& dataset) {
    for (float*& ptr : dataset.cluster_ptrs) {
        if (ptr) {
            cudaFreeHost(ptr);
            ptr = nullptr;
        }
    }
    dataset.cluster_ptrs.clear();
    dataset.cluster_sizes.clear();
}

static BalancedMapping build_balanced_mapping(const ClusterDataset& dataset,
                                              int chunk_size) {
    BalancedMapping mapping;
    int n_clusters = static_cast<int>(dataset.cluster_sizes.size());
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

static void release_query_batch(QueryBatch& batch) {
    if (batch.ptrs) {
        free_vector_list(reinterpret_cast<void**>(batch.ptrs));
        batch.ptrs = nullptr;
    }
}

// 余弦距离：1 - cos，相比 L2 更符合当前测试目标
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

static QueryBatch generate_queries(const BenchmarkCase& config, std::mt19937& rng) {
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
    auto ptrs = reinterpret_cast<float**>(
        malloc_vector_list(config.n_query, config.vector_dim, sizeof(float)));
    if (!ptrs) {
        throw std::runtime_error("malloc_vector_list failed for queries");
    }
    for (int qi = 0; qi < config.n_query; ++qi) {
        for (int d = 0; d < config.vector_dim; ++d) {
            ptrs[qi][d] = value_dist(rng);
        }
    }
    QueryBatch batch;
    batch.ptrs = ptrs;
    return batch;
}

static std::vector<float> generate_cluster_centers(const BenchmarkCase& config,
                                                   std::mt19937& rng) {
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
    std::vector<float> centers(static_cast<size_t>(config.n_clusters) * config.vector_dim);
    for (float& v : centers) {
        v = value_dist(rng);
    }
    return centers;
}

static void cpu_coarse_fine_search(const BenchmarkCase& config,
                                   const QueryBatch& queries,
                                   const ClusterDataset& dataset,
                                   const std::vector<float>& centers,
                                   int n_cluster_per_query,
                                   int topn,
                                   std::vector<int>& out_index,
                                   std::vector<float>& out_dist) {
    const int n_query = config.n_query;
    const int n_dim = config.vector_dim;
    const int n_clusters = config.n_clusters;

    struct Pair { float dist; int idx; };
    std::vector<int> coarse_indices(static_cast<size_t>(n_query) * n_cluster_per_query);

    const unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());
    auto worker = [&](int start, int end) {
        std::vector<Pair> tmp(n_clusters);
        std::vector<Pair> fine_buffer;
        for (int qi = start; qi < end; ++qi) {
            const float* query = queries.ptrs[qi];

            // coarse: compare with cluster centers
            for (int cid = 0; cid < n_clusters; ++cid) {
                const float* center = centers.data() + static_cast<size_t>(cid) * n_dim;
                tmp[cid] = {cosine_distance(query, center, n_dim), cid};
            }
            std::partial_sort(tmp.begin(), tmp.begin() + n_cluster_per_query, tmp.end(),
                              [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            for (int k = 0; k < n_cluster_per_query; ++k) {
                coarse_indices[static_cast<size_t>(qi) * n_cluster_per_query + k] = tmp[k].idx;
            }

            // fine: iterate vectors inside selected clusters
            fine_buffer.clear();
            for (int k = 0; k < n_cluster_per_query; ++k) {
                int cid = coarse_indices[static_cast<size_t>(qi) * n_cluster_per_query + k];
                int vec_count = dataset.cluster_sizes[cid];
                const float* base = dataset.cluster_ptrs[cid];
                for (int vid = 0; vid < vec_count; ++vid) {
                    const float* vec = base + static_cast<size_t>(vid) * n_dim;
                    fine_buffer.push_back({cosine_distance(query, vec, n_dim), vid});
                }
            }
            if (fine_buffer.size() > static_cast<size_t>(topn)) {
                std::nth_element(fine_buffer.begin(), fine_buffer.begin() + topn, fine_buffer.end(),
                                 [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
                fine_buffer.resize(topn);
            } else {
                std::sort(fine_buffer.begin(), fine_buffer.end(),
                          [](const Pair& a, const Pair& b) { return a.dist < b.dist; });
            }
            std::sort(fine_buffer.begin(), fine_buffer.end(),
                      [](const Pair& a, const Pair& b) { return a.dist < b.dist; });

            for (int t = 0; t < topn; ++t) {
                size_t offset = static_cast<size_t>(qi) * topn + t;
                if (t < static_cast<int>(fine_buffer.size())) {
                    out_index[offset] = fine_buffer[t].idx;
                    out_dist[offset] = fine_buffer[t].dist;
                } else {
                    out_index[offset] = -1;
                    out_dist[offset] = 0.0f;
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

static void run_case(const BenchmarkCase& config) {
    std::mt19937 rng(1234);

    auto dataset = prepare_cluster_dataset(config, rng);
    QueryBatch queries = generate_queries(config, rng);
    auto centers = generate_cluster_centers(config, rng);
    auto balanced = build_balanced_mapping(dataset, 512);

    float** cluster_data = dataset.cluster_ptrs.data();
    int* cluster_sizes = dataset.cluster_sizes.data();
    float** query_ptrs = queries.ptrs;
    float* cluster_center_data = centers.data();

    auto run_pipeline = [&](bool use_balance, const char* tag, std::vector<int>& out_idx,
                            std::vector<float>& out_dist) {
        float** topk_dist = reinterpret_cast<float**>(
            malloc_vector_list(config.n_query, config.topn, sizeof(float)));
        int** topk_index = reinterpret_cast<int**>(
            malloc_vector_list(config.n_query, config.topn, sizeof(int)));
        int* n_isnull = new int[config.n_query];

        auto start = std::chrono::high_resolution_clock::now();
        batch_search_pipeline(
            query_ptrs,
            cluster_sizes,
            cluster_data,
            cluster_center_data,
            topk_dist,
            topk_index,
            n_isnull,
            config.n_query,
            config.vector_dim,
            config.n_clusters,
            std::min(config.n_clusters, config.topk),
            config.topn,
            use_balance,
            use_balance ? balanced.cluster2block_offset.data() : nullptr,
            use_balance ? balanced.cluster2block_ids.data() : nullptr,
            use_balance ? balanced.cluster2block_local_offsets.data() : nullptr,
            use_balance ? balanced.block_vector_counts.data() : nullptr,
            use_balance ? balanced.block_count : 0
        );
        auto end = std::chrono::high_resolution_clock::now();
        double ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        COUT_VAL("[", tag, "] n_query=", config.n_query,
                 " clusters=", config.n_clusters,
                 " dim=", config.vector_dim,
                 " elapsed=", ms, "ms");
        COUT_VAL("  sample query0 isnull=", n_isnull[0],
                 " top_index[0]=", topk_index[0][0],
                 " top_dist[0]=", topk_dist[0][0]);

        for (int qi = 0; qi < config.n_query; ++qi) {
            for (int t = 0; t < config.topn; ++t) {
                size_t offset = static_cast<size_t>(qi) * config.topn + t;
                out_idx[offset] = topk_index[qi][t];
                out_dist[offset] = topk_dist[qi][t];
            }
        }

        free_vector_list(reinterpret_cast<void**>(topk_dist));
        free_vector_list(reinterpret_cast<void**>(topk_index));
        delete[] n_isnull;
    };

    std::vector<int> gpu_idx_unbalanced(static_cast<size_t>(config.n_query) * config.topn);
    std::vector<float> gpu_dist_unbalanced(static_cast<size_t>(config.n_query) * config.topn);
    std::vector<int> gpu_idx_balanced = gpu_idx_unbalanced;
    std::vector<float> gpu_dist_balanced = gpu_dist_unbalanced;

    run_pipeline(false, "unbalanced", gpu_idx_unbalanced, gpu_dist_unbalanced);
    run_pipeline(true, "balanced", gpu_idx_balanced, gpu_dist_balanced);

    std::vector<int> cpu_idx(static_cast<size_t>(config.n_query) * config.topn);
    std::vector<float> cpu_dist(static_cast<size_t>(config.n_query) * config.topn);
    cpu_coarse_fine_search(config, queries, dataset, centers,
                           std::min(config.n_clusters, config.topk),
                           config.topn,
                           cpu_idx,
                           cpu_dist);

    auto compare_results = [&](const char* tag,
                               const std::vector<int>& gpu_idx,
                               const std::vector<float>& gpu_dist) {
        int mismatch = 0;
        for (size_t i = 0; i < cpu_idx.size(); ++i) {
            if (gpu_idx[i] != cpu_idx[i] ||
                std::fabs(gpu_dist[i] - cpu_dist[i]) > 1e-3f) {
                ++mismatch;
            }
        }
        COUT_VAL("[compare] ", tag, " mismatches=", mismatch);
    };

    compare_results("unbalanced", gpu_idx_unbalanced, gpu_dist_unbalanced);
    compare_results("balanced", gpu_idx_balanced, gpu_dist_balanced);

    release_query_batch(queries);
    release_cluster_dataset(dataset);
}

int main() {
    std::vector<BenchmarkCase> cases = {
        {512, 64, 128, 32, 32},
        {2000, 256, 256, 64, 64},
        {4000, 512, 512, 128, 128}
    };

    for (const auto& c : cases) {
        run_case(c);
    }
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    return 0;
}
