#include "../pch.h"
#include "integrate_screen.cuh"

#include "../fusion_cos_topk/fusion_cos_topk.cuh"
#include "../fine_screen_top_n/fine_screen_top_n.cuh"
#include "../cudatimer.h"
#include "../../unit_tests/common/test_utils.cuh"

#include <algorithm>
#include <cstring>
#include <cfloat>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

void batch_search_pipeline(float** query_batch,
                           int* cluster_size,
                           float*** cluster_data,
                           float** cluster_center_data,
                           
                           float** topk_dist,
                           int** topk_index,
                           int* n_isnull,

                           int** coarse_indices,
                           float** coarse_dists,

                           int n_query,
                           int n_dim,
                           int n_total_cluster,
                           int n_probes,
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
    if (n_probes <= 0 || n_probes > n_total_cluster) {
        throw std::invalid_argument("invalid n_probes");
    }

    // std::vector<float> h_query_flat;
    // flatten_query_batch(query_batch, n_query, n_dim, h_query_flat);

    float* d_queries = nullptr;
    float** d_clusters = (float**)malloc(n_total_cluster * sizeof(float*));

    auto cleanup = [&]() {
        if (d_queries) {
            cudaFree(d_queries);
            d_queries = nullptr;
        }
        if (d_clusters) {
            for (int i = 0; i < n_total_cluster; ++i) {
                if (d_clusters[i]) {
                    cudaFree(d_clusters[i]);
                    d_clusters[i] = nullptr;
                }
            }
            free(d_clusters);
            d_clusters = nullptr;
        }
    };

    try {
        cudaMalloc(&d_queries, n_query * n_dim * sizeof(float));
        cudaMemcpy(d_queries, query_batch[0], n_query * n_dim * sizeof(float), cudaMemcpyHostToDevice);

        for (int cluster_id = 0; cluster_id < n_total_cluster; ++cluster_id) {
            cudaMalloc(&d_clusters[cluster_id], cluster_size[cluster_id] * n_dim * sizeof(float));
            cudaMemcpy(d_clusters[cluster_id],
                       cluster_data[cluster_id][0],
                       cluster_size[cluster_id] * n_dim * sizeof(float),
                       cudaMemcpyHostToDevice);
        }
        CHECK_CUDA_ERRORS

        // ------------------------------------------------------------------
        // Step 1. 粗筛：调用 warpsort 融合算子，得到 query -> cluster mapping
        // ------------------------------------------------------------------
        // 注意：data_index 在 cuda_cos_topk_warpsort 内部使用 CUDA kernel 自动生成顺序索引 [0, 1, 2, ..., n_total_cluster-1]
        {
            CUDATimer timer("Step 1: Coarse Search (cuda_cos_topk_warpsort)");
            cuda_cos_topk_warpsort(
                query_batch,
                cluster_center_data,
                coarse_indices,
                coarse_dists,
                n_query,
                n_total_cluster,
                n_dim,
                n_probes
            );
            cudaDeviceSynchronize();
            CHECK_CUDA_ERRORS
        }
        
        // // 输出GPU粗筛结果（仅在小规模测试时）
        // if (n_query <= 4) {
        //     printf("=== GPU Coarse Search Results ===\n");
        //     for (int qi = 0; qi < n_query; ++qi) {
        //         printf("Query %d coarse clusters: ", qi);
        //         for (int k = 0; k < n_probes; ++k) {
        //             printf("(cluster=%d dist=%.6f) ", 
        //                    coarse_indices[qi][k], 
        //                    coarse_dists[qi][k]);
        //         }
        //         printf("\n");
        //     }
        // }

        // ------------------------------------------------------------------
        // Step 2. 将 query→cluster 粗筛结果转成 block 序列
        // 同时构建 query->probe 映射和 probe 在 query 中的索引
        // ------------------------------------------------------------------
        std::vector<float*> compact_block_host_ptrs;
        std::vector<int> compact_block_sizes;
        int active_block_count = 0;
        int* block_query_offset;
        int* block_query_data;
        int* block_query_probe_indices;  // 新增：每个block-query对中probe在query中的索引

        {
            CUDATimer timer("Step 2: Convert query→cluster to block sequence", true, false);
            std::unordered_map<int, int> block_id_to_compact;
            std::vector<int> compact_block_ids;
            std::vector<std::vector<int>> block_to_queries;
            std::vector<std::vector<int>> block_query_probe_indices_vec;  // 新增：存储probe索引

            // 构建 query->probe 映射（用于计算 probe 在 query 中的索引）
            std::vector<std::vector<int>> query_probe_list(n_query);
            std::vector<std::unordered_map<int, int>> query_probe_index_map(n_query);  // query_id -> (probe_id -> index)

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
                    block_query_probe_indices_vec.emplace_back();  // 新增
                    return static_cast<int>(compact_block_ids.size()) - 1;
                }
                return it->second;
            };

            for (int qi = 0; qi < n_query; ++qi) {
                // const int* cluster_list = coarse_indices[qi];
                for (int rank = 0; rank < n_probes; ++rank) {
                    int cluster_id = coarse_indices[qi][rank];


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
                            float* cluster_base = cluster_data[cluster_id][0];
                            if (!cluster_base) {
                                throw std::runtime_error("cluster data pointer is null");
                            }
                            float* block_ptr = cluster_base + static_cast<size_t>(local_offset) * n_dim;
                            int vec_count = block_vector_counts[block_id];
                            int compact_idx = acquire_block_slot(block_id, block_ptr, vec_count);
                            
                            // 计算probe在query中的索引
                            int probe_index_in_query = -1;
                            auto& probe_map = query_probe_index_map[qi];
                            auto map_it = probe_map.find(compact_idx);
                            if (map_it == probe_map.end()) {
                                // 第一次遇到这个probe，添加到query的probe列表
                                probe_index_in_query = static_cast<int>(query_probe_list[qi].size());
                                query_probe_list[qi].push_back(compact_idx);
                                probe_map[compact_idx] = probe_index_in_query;
                            } else {
                                // 已经存在，使用已有索引
                                probe_index_in_query = map_it->second;
                            }
                            
                            block_to_queries[compact_idx].push_back(qi);
                            block_query_probe_indices_vec[compact_idx].push_back(probe_index_in_query);
                        }
                    }


                    else {
                        float* block_ptr = cluster_data[cluster_id][0];
                        if (!block_ptr) {
                            throw std::runtime_error("cluster data pointer is null");
                        }
                        int vec_count = cluster_size[cluster_id];
                        int compact_idx = acquire_block_slot(cluster_id, block_ptr, vec_count);
                        
                        // 计算probe在query中的索引
                        int probe_index_in_query = -1;
                        auto& probe_map = query_probe_index_map[qi];
                        auto map_it = probe_map.find(compact_idx);
                        if (map_it == probe_map.end()) {
                            // 第一次遇到这个probe，添加到query的probe列表
                            probe_index_in_query = static_cast<int>(query_probe_list[qi].size());
                            query_probe_list[qi].push_back(compact_idx);
                            probe_map[compact_idx] = probe_index_in_query;
                        } else {
                            // 已经存在，使用已有索引
                            probe_index_in_query = map_it->second;
                        }
                        
                        block_to_queries[compact_idx].push_back(qi);
                        block_query_probe_indices_vec[compact_idx].push_back(probe_index_in_query);
                    }
                }
            }

            active_block_count = static_cast<int>(compact_block_ids.size());
            block_query_offset = (int*)malloc((active_block_count + 1) * sizeof(int));

            int total_entries = 0;
            for (int block_idx = 0; block_idx < active_block_count; ++block_idx) {
                block_query_offset[block_idx] = total_entries;
                total_entries += block_to_queries[block_idx].size();
            }
            block_query_offset[active_block_count] = total_entries;
            block_query_data = (int*)malloc(total_entries * sizeof(int));
            block_query_probe_indices = (int*)malloc(total_entries * sizeof(int));  // 新增
            int cursor = 0;
            for (int block_idx = 0; block_idx < active_block_count; ++block_idx) {
                const auto& qlist = block_to_queries[block_idx];
                const auto& probe_indices = block_query_probe_indices_vec[block_idx];
                memcpy(block_query_data + cursor, qlist.data(), qlist.size() * sizeof(int));
                memcpy(block_query_probe_indices + cursor, probe_indices.data(), probe_indices.size() * sizeof(int));  // 新增
                cursor += qlist.size();
            }
            
            // 输出 Step 2 结果，验证是否由粗筛结果转换而来
            if (n_query <= 4 && n_probes <= 8) {
                // printf("\n=== Step 2: Block Sequence Conversion Results ===\n");
                
                // // 1. 输出粗筛结果（输入）
                // printf("--- Coarse Search Results (Input) ---\n");
                // for (int qi = 0; qi < n_query; ++qi) {
                //     printf("Query %d selected clusters: ", qi);
                //     for (int rank = 0; rank < n_probes; ++rank) {
                //         int cluster_id = coarse_indices[qi][rank];
                //         if (cluster_id >= 0 && cluster_id < n_total_cluster) {
                //             printf("%d ", cluster_id);
                //         } else {
                //             printf("(invalid:%d) ", cluster_id);
                //         }
                //     }
                //     printf("\n");
                // }
                
                // // 2. 输出 Step 2 生成的 block 列表和对应的 query 列表
                // printf("\n--- Step 2 Output: Block -> Query Mapping ---\n");
                // printf("Total active blocks: %d\n", active_block_count);
                // for (int block_idx = 0; block_idx < active_block_count; ++block_idx) {
                //     int global_block_id = compact_block_ids[block_idx];
                //     const auto& qlist = block_to_queries[block_idx];
                //     printf("Block[%d] (global_id=%d, vec_count=%d) -> queries: [", 
                //            block_idx, global_block_id, compact_block_sizes[block_idx]);
                //     for (size_t i = 0; i < qlist.size(); ++i) {
                //         printf("%d", qlist[i]);
                //         if (i < qlist.size() - 1) printf(", ");
                //     }
                //     printf("] (%zu queries)\n", qlist.size());
                // }
                
                // // 3. 验证：检查每个 query 的 cluster 是否都对应到了 block
                // printf("\n--- Verification: Query -> Block Mapping ---\n");

                bool all_match = true;
                for (int qi = 0; qi < n_query; ++qi) {
                    // printf("Query %d: ", qi);
                    std::vector<int> expected_blocks;
                    for (int rank = 0; rank < n_probes; ++rank) {
                        int cluster_id = coarse_indices[qi][rank];
                        if (cluster_id < 0 || cluster_id >= n_total_cluster) continue;
                        
                        if (use_balanced_blocks) {
                            int start = cluster2block_offset[cluster_id];
                            int end = cluster2block_offset[cluster_id + 1];
                            for (int idx = start; idx < end; ++idx) {
                                int block_id = cluster2block_ids[idx];
                                if (block_id >= 0 && block_id < n_blocks) {
                                    expected_blocks.push_back(block_id);
                                }
                            }
                        } else {
                            expected_blocks.push_back(cluster_id);
                        }
                    }
                    
                    // // 检查这些 block 是否都在 Step 2 的输出中，并且包含 query qi
                    // printf("expected blocks: [");
                    // for (size_t i = 0; i < expected_blocks.size(); ++i) {
                    //     printf("%d", expected_blocks[i]);
                    //     if (i < expected_blocks.size() - 1) printf(", ");
                    // }
                    // printf("] -> ");
                    
                    std::vector<int> found_blocks;
                    for (int block_idx = 0; block_idx < active_block_count; ++block_idx) {
                        const auto& qlist = block_to_queries[block_idx];
                        if (std::find(qlist.begin(), qlist.end(), qi) != qlist.end()) {
                            found_blocks.push_back(compact_block_ids[block_idx]);
                        }
                    }
                    // printf("found in blocks: [");
                    // for (size_t i = 0; i < found_blocks.size(); ++i) {
                    //     printf("%d", found_blocks[i]);
                    //     if (i < found_blocks.size() - 1) printf(", ");
                    // }
                    // printf("]");
                    
                    // 验证是否匹配
                    if (expected_blocks.size() != found_blocks.size()) {
                        all_match = false;
                    } else {
                        std::sort(expected_blocks.begin(), expected_blocks.end());
                        std::sort(found_blocks.begin(), found_blocks.end());
                        all_match &= (expected_blocks == found_blocks);
                    }
                }
                printf("step 2 all_match: %s\n", all_match ? "true" : "false");
            }
        }

        COUT_ENDL("n_query: ", n_query, "n_probes: ", n_probes, "k: ", k);
        int** candidate_index = (int**)malloc_vector_list(n_query, n_probes * k, sizeof(int));
        float** candidate_dist = (float**)malloc_vector_list(n_query, n_probes * k, sizeof(float));
        // table_2D("candidate_index", candidate_index, n_query, n_probes * k);
        // table_2D("candidate_dist", candidate_dist, n_query, n_probes * k);

        // ------------------------------------------------------------------
        // Step 3. 精筛：上传 block 向量 + block 查询映射，调用 GPU kernel
        // ------------------------------------------------------------------
        COUT_ENDL("Step 3: Fine Search (fine_screen_top_n_blocks)");
        if (active_block_count > 0) {
            {
                CUDATimer timer("Step 3: Fine Search (fine_screen_top_n_blocks)");
                // 调用固定probe版本的精筛
                fine_screen_top_n_blocks(
                    query_batch[0],

                    compact_block_host_ptrs.data(),
                    compact_block_sizes.data(),
                    block_query_offset,
                    block_query_data,
                    block_query_probe_indices,  // 新增：probe在query中的索引

                    topk_index[0],
                    topk_dist[0],
                
                    candidate_dist,
                    candidate_index,

                    n_query,
                    active_block_count,
                    n_dim,
                    k
                );
                CHECK_CUDA_ERRORS
            }
            
            // 释放Step 2中分配的内存
            if (block_query_offset) {
                free(block_query_offset);
                block_query_offset = nullptr;
            }
            if (block_query_data) {
                free(block_query_data);
                block_query_data = nullptr;
            }
            if (block_query_probe_indices) {
                free(block_query_probe_indices);
                block_query_probe_indices = nullptr;
            }

        // 输出GPU精筛结果
        if (n_query <= 4) {
            printf("=== GPU Fine Search Results ===\n");
            for (int qi = 0; qi < n_query; ++qi) {
                printf("Query %d: ", qi);
                for (int t = 0; t < k; ++t) {
                    printf("(idx=%d, dist=%.6f) ", 
                           topk_index[qi][t],
                           topk_dist[qi][t]);
                }
                printf("\n");
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
        
        // 正常执行完成，清理资源
        cleanup();
    } catch (...) {
        // 异常发生时，清理已分配的资源
        cleanup();
        throw;
    }
}

void run_integrate_pipeline() {
    // TODO: 后续补充粗筛 + 精筛整体调度
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS
}
