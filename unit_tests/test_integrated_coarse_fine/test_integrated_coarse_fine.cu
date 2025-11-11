#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "../../cuda/integrate_screen/integrate_screen.cuh"
#include "../common/test_utils.cuh"

#define CUDA_CHECK(cmd)                                                   \
    do {                                                                  \
        cudaError_t _err = (cmd);                                         \
        if (_err != cudaSuccess) {                                        \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(_err)      \
                      << " @ " << __FILE__ << ":" << __LINE__ << "\n";    \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

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
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_ptr), bytes));

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
    std::vector<std::vector<float>> storage;
    std::vector<float*> ptrs;
};

static QueryBatch generate_queries(const BenchmarkCase& config, std::mt19937& rng) {
    std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
    QueryBatch batch;
    batch.storage.resize(config.n_query, std::vector<float>(config.vector_dim, 0.0f));
    batch.ptrs.resize(config.n_query);

    for (int qi = 0; qi < config.n_query; ++qi) {
        for (int d = 0; d < config.vector_dim; ++d) {
            batch.storage[qi][d] = value_dist(rng);
        }
        batch.ptrs[qi] = batch.storage[qi].data();
    }
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

static void run_case(const BenchmarkCase& config) {
    std::mt19937 rng(1234);

    auto dataset = prepare_cluster_dataset(config, rng);
    auto queries = generate_queries(config, rng);
    auto centers = generate_cluster_centers(config, rng);
    auto balanced = build_balanced_mapping(dataset, 512);

    float** cluster_data = dataset.cluster_ptrs.data();
    int* cluster_sizes = dataset.cluster_sizes.data();
    float** query_ptrs = queries.ptrs.data();
    float* cluster_center_data = centers.data();

    auto run_pipeline = [&](bool use_balance, const char* tag) {
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

        free_vector_list(reinterpret_cast<void**>(topk_dist));
        free_vector_list(reinterpret_cast<void**>(topk_index));
        delete[] n_isnull;
    };

    run_pipeline(false, "unbalanced");
    run_pipeline(true, "balanced");

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
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
