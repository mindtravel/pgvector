#include "../pch.h"
#include "integrate_screen.cuh"

#include "../fusion_cos_topk/fusion_cos_topk.cuh"
#include "../fine_screen_top_n/fine_screen_top_n.cuh"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

inline void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("[cuda error] ") + expr + " -> " + cudaGetErrorString(status) +
            " (" + file + ":" + std::to_string(line) + ")");
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

inline void flatten_query_batch(float** query_batch,
                                int n_query,
                                int n_dim,
                                std::vector<float>& buffer) {
    buffer.resize(static_cast<size_t>(n_query) * n_dim);
    for (int i = 0; i < n_query; ++i) {
        if (!query_batch || !query_batch[i]) {
            throw std::invalid_argument("query_batch contains null pointer");
        }
        std::memcpy(&buffer[static_cast<size_t>(i) * n_dim],
                    query_batch[i],
                    static_cast<size_t>(n_dim) * sizeof(float));
    }
}

}  // namespace

void batch_search_pipeline(float** query_batch,
                           int* cluster_size,
                           float** cluster_data,
                           float* cluster_center_data,
                           float** topk_dist,
                           int** topk_index,
                           int* n_isnull,
                           int n_query,
                           int n_dim,
                           int n_total_cluster,
                           int n_cluster_per_query,
                           int k,
                           bool use_balanced_blocks,
                           const int* cluster2block_offset,
                           const int* cluster2block_ids,
                           const int* cluster2block_local_offsets,
                           const int* block_vector_counts,
                           int n_blocks) {

    if (n_query <= 0 || n_dim <= 0 || n_total_cluster <= 0 || k <= 0) {
        throw std::invalid_argument("invalid batch_search_pipeline configuration");
    }
    if (!cluster_size || !cluster_data) {
        throw std::invalid_argument("cluster metadata is null");
    }
    if (use_balanced_blocks) {
        if (!cluster2block_offset || !cluster2block_ids || !cluster2block_local_offsets ||
            !block_vector_counts) {
            throw std::invalid_argument("balanced-block inputs are required when use_balanced_blocks=true");
        }
        if (n_blocks <= 0) {
            throw std::invalid_argument("balanced block count must be positive");
        }
    }

    if (!cluster_center_data) {
        throw std::invalid_argument("cluster_center_data must not be null for coarse search");
    }
    if (n_cluster_per_query <= 0 || n_cluster_per_query > n_total_cluster) {
        throw std::invalid_argument("invalid n_cluster_per_query");
    }

    std::vector<float> h_query_flat;
    flatten_query_batch(query_batch, n_query, n_dim, h_query_flat);

    float* d_queries = nullptr;
    std::vector<float*> d_clusters(n_total_cluster, nullptr);

    auto cleanup = [&]() {
        if (d_queries) {
            cudaFree(d_queries);
            d_queries = nullptr;
        }
        for (float*& ptr : d_clusters) {
            if (ptr) {
                cudaFree(ptr);
                ptr = nullptr;
            }
        }
    };

    try {
        const size_t query_bytes = h_query_flat.size() * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_queries, query_bytes));
        CUDA_CHECK(cudaMemcpy(d_queries, h_query_flat.data(), query_bytes, cudaMemcpyHostToDevice));

        for (int cluster_id = 0; cluster_id < n_total_cluster; ++cluster_id) {
            const int vec_count = cluster_size[cluster_id];
            if (vec_count <= 0 || !cluster_data[cluster_id]) {
                continue;
            }
            const size_t bytes = static_cast<size_t>(vec_count) * n_dim * sizeof(float);
            CUDA_CHECK(cudaMalloc(&d_clusters[cluster_id], bytes));
            CUDA_CHECK(cudaMemcpy(d_clusters[cluster_id],
                                  cluster_data[cluster_id],
                                  bytes,
                                  cudaMemcpyHostToDevice));
        }

        // ------------------------------------------------------------------
        // Step 1. 粗筛：调用 warpsort 融合算子，得到 query -> cluster mapping
        // ------------------------------------------------------------------
        std::vector<float*> query_ptrs(n_query);
        for (int qi = 0; qi < n_query; ++qi) {
            if (!query_batch || !query_batch[qi]) {
                throw std::invalid_argument("query_batch contains null pointer");
            }
            query_ptrs[qi] = query_batch[qi];
        }

        std::vector<float*> cluster_center_ptrs(n_total_cluster);
        for (int cid = 0; cid < n_total_cluster; ++cid) {
            cluster_center_ptrs[cid] = cluster_center_data + static_cast<size_t>(cid) * n_dim;
        }

        // data_index 描述每个 query 对 n_total_cluster 个候选的全局 ID
        std::vector<int> data_index_storage(static_cast<size_t>(n_query) * n_total_cluster);
        std::vector<int*> data_index_ptrs(n_query);
        for (int qi = 0; qi < n_query; ++qi) {
            data_index_ptrs[qi] = data_index_storage.data() + static_cast<size_t>(qi) * n_total_cluster;
            for (int cid = 0; cid < n_total_cluster; ++cid) {
                data_index_ptrs[qi][cid] = cid;
            }
        }

        std::vector<int> coarse_index_storage(static_cast<size_t>(n_query) * n_cluster_per_query);
        std::vector<int*> coarse_index_ptrs(n_query);
        for (int qi = 0; qi < n_query; ++qi) {
            coarse_index_ptrs[qi] = coarse_index_storage.data() + static_cast<size_t>(qi) * n_cluster_per_query;
        }

        std::vector<float> coarse_dist_storage(static_cast<size_t>(n_query) * n_cluster_per_query, 0.0f);
        std::vector<float*> coarse_dist_ptrs(n_query);
        for (int qi = 0; qi < n_query; ++qi) {
            coarse_dist_ptrs[qi] = coarse_dist_storage.data() + static_cast<size_t>(qi) * n_cluster_per_query;
        }

        cuda_cos_topk_warpsort(
            query_ptrs.data(),
            cluster_center_ptrs.data(),
            data_index_ptrs.data(),
            coarse_index_ptrs.data(),
            coarse_dist_ptrs.data(),
            n_query,
            n_total_cluster,
            n_dim,
            n_cluster_per_query
        );

        // ------------------------------------------------------------------
        // Step 2. 将 query→cluster 粗筛结果转成 block 序列
        // ------------------------------------------------------------------
        std::unordered_map<int, int> block_id_to_compact;
        std::vector<int> compact_block_ids;
        std::vector<float*> compact_block_host_ptrs;
        std::vector<int> compact_block_sizes;
        std::vector<std::vector<int>> block_to_queries;

        auto acquire_block_slot = [&](int global_block_id,
                                      float* host_ptr,
                                      int vec_count) -> int {
            auto [it, inserted] =
                block_id_to_compact.emplace(global_block_id,
                                            static_cast<int>(compact_block_ids.size()));
            if (inserted) {
                compact_block_ids.push_back(global_block_id);
                compact_block_host_ptrs.push_back(host_ptr);
                compact_block_sizes.push_back(vec_count);
                block_to_queries.emplace_back();
                return static_cast<int>(compact_block_ids.size()) - 1;
            }
            return it->second;
        };

        for (int qi = 0; qi < n_query; ++qi) {
            const int* cluster_list = coarse_index_ptrs[qi];
            for (int rank = 0; rank < n_cluster_per_query; ++rank) {
                int cluster_id = cluster_list[rank];
                if (cluster_id < 0 || cluster_id >= n_total_cluster) continue;
                if (use_balanced_blocks) {
                    int start = cluster2block_offset[cluster_id];
                    int end = cluster2block_offset[cluster_id + 1];
                    for (int idx = start; idx < end; ++idx) {
                        int block_id = cluster2block_ids[idx];
                        if (block_id < 0 || block_id >= n_blocks) {
                            throw std::out_of_range("balanced block id out of range");
                        }
                        int local_offset = cluster2block_local_offsets[idx];
                        float* cluster_base = cluster_data[cluster_id];
                        if (!cluster_base) {
                            throw std::runtime_error("cluster data pointer is null");
                        }
                        float* block_ptr = cluster_base + static_cast<size_t>(local_offset) * n_dim;
                        int vec_count = block_vector_counts[block_id];
                        int compact_idx = acquire_block_slot(block_id, block_ptr, vec_count);
                        block_to_queries[compact_idx].push_back(qi);
                    }
                } else {
                    float* block_ptr = cluster_data[cluster_id];
                    if (!block_ptr) {
                        throw std::runtime_error("cluster data pointer is null");
                    }
                    int vec_count = cluster_size[cluster_id];
                    int compact_idx = acquire_block_slot(cluster_id, block_ptr, vec_count);
                    block_to_queries[compact_idx].push_back(qi);
                }
            }
        }

        const int active_block_count = static_cast<int>(compact_block_ids.size());
        std::vector<int> block_query_offset(active_block_count + 1, 0);
        std::vector<int> block_query_data;
        block_query_data.reserve(static_cast<size_t>(n_query) * n_cluster_per_query);

        size_t total_entries = 0;
        for (int block_idx = 0; block_idx < active_block_count; ++block_idx) {
            block_query_offset[block_idx] = static_cast<int>(total_entries);
            total_entries += block_to_queries[block_idx].size();
        }
        block_query_offset[active_block_count] = static_cast<int>(total_entries);
        block_query_data.resize(total_entries);
        size_t cursor = 0;
        for (int block_idx = 0; block_idx < active_block_count; ++block_idx) {
            const auto& qlist = block_to_queries[block_idx];
            std::copy(qlist.begin(), qlist.end(), block_query_data.begin() + cursor);
            cursor += qlist.size();
        }

        // ------------------------------------------------------------------
        // Step 3. 精筛：上传 block 向量 + block 查询映射，调用 GPU kernel
        // ------------------------------------------------------------------
        if (active_block_count > 0) {
            std::vector<int> fine_topn_index(static_cast<size_t>(n_query) * k);
            std::vector<float> fine_topn_dist(static_cast<size_t>(n_query) * k);

            fine_screen_top_n_blocks(
                h_query_flat.data(),
                n_query,
                n_dim,
                k,
                compact_block_host_ptrs.data(),
                compact_block_sizes.data(),
                active_block_count,
                block_query_offset.data(),
                block_query_data.data(),
                fine_topn_index.data(),
                fine_topn_dist.data()
            );

            if (topk_index) {
                for (int qi = 0; qi < n_query; ++qi) {
                    if (topk_index[qi]) {
                        std::copy_n(&fine_topn_index[static_cast<size_t>(qi) * k], k, topk_index[qi]);
                    }
                }
            }
            if (topk_dist) {
                for (int qi = 0; qi < n_query; ++qi) {
                    if (topk_dist[qi]) {
                        std::copy_n(&fine_topn_dist[static_cast<size_t>(qi) * k], k, topk_dist[qi]);
                    }
                }
            }
            if (n_isnull) {
                std::fill(n_isnull, n_isnull + n_query, 0);
            }
        } else {
            for (int qi = 0; qi < n_query; ++qi) {
                if (n_isnull) n_isnull[qi] = k;
                if (topk_dist && topk_dist[qi]) {
                    std::fill(topk_dist[qi], topk_dist[qi] + k, 0.0f);
                }
                if (topk_index && topk_index[qi]) {
                    std::fill(topk_index[qi], topk_index[qi] + k, -1);
                }
            }
        }
    } catch (...) {
        cleanup();
        throw;
    }

    cleanup();
}

void run_integrate_pipeline() {
    // TODO: 后续补充粗筛 + 精筛整体调度
    CUDA_CHECK(cudaDeviceSynchronize());
}
