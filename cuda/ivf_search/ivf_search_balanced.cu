#include "../pch.h"
#include "ivf_search.cuh"

#include "../fusion_dist_topk/fusion_dist_topk.cuh"
#include "../fine_screen_top_n/fine_screen_top_n.cuh"
#include "../cudatimer.h"
#include "../utils.cuh"

#include <algorithm>
#include <cstring>
#include <cfloat>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

void ivf_search_pipeline(float** query_batch,
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
        throw std::invalid_argument("invalid ivf_search_pipeline configuration");
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
        // 使用纯C数组实现，不使用C++容器
        // ------------------------------------------------------------------
        float** compact_block_host_ptrs = nullptr;
        int* compact_block_sizes = nullptr;
        int* compact_block_ids = nullptr;
        int active_block_count = 0;
        int* block_query_offset = nullptr;
        int* block_query_data = nullptr;
        int* block_query_probe_indices = nullptr;  // 每个block-query对中probe在query中的索引

        {
            CUDATimer timer("Step 2: Convert query→cluster to block sequence", true, false);
            
            // 确定最大可能的block数量
            int max_blocks = use_balanced_blocks ? n_blocks : n_total_cluster;
            
            // 第一步：收集所有query-block对，同时记录block信息
            // 使用临时数组记录每个block的信息（是否已出现、host_ptr、vec_count、global_id）
            int* block_seen = (int*)calloc(max_blocks, sizeof(int));  // 0=未出现，1=已出现
            float** block_host_ptrs = (float**)calloc(max_blocks, sizeof(float*));
            int* block_vec_counts = (int*)calloc(max_blocks, sizeof(int));
            int* block_global_ids = (int*)calloc(max_blocks, sizeof(int));
            
            // 临时数组：记录每个block对应的query（用于后续构建CSR）
            int max_pairs = n_query * n_probes * (use_balanced_blocks ? 4 : 1);  // 估算最大对数
            int* temp_block_ids = (int*)malloc(max_pairs * sizeof(int));
            int* temp_query_ids = (int*)malloc(max_pairs * sizeof(int));
            int temp_pair_count = 0;
            
            // 遍历所有query-cluster对，收集block信息
            for (int qi = 0; qi < n_query; ++qi) {
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
                            
                            // 记录block信息（如果第一次遇到）
                            if (!block_seen[block_id]) {
                                block_seen[block_id] = 1;
                                int local_offset = cluster2block_local_offsets[idx];
                                float* cluster_base = cluster_data[cluster_id][0];
                                if (!cluster_base) {
                                    throw std::runtime_error("cluster data pointer is null");
                                }
                                block_host_ptrs[block_id] = cluster_base + static_cast<size_t>(local_offset) * n_dim;
                                block_vec_counts[block_id] = block_vector_counts[block_id];
                                block_global_ids[block_id] = block_id;
                            }
                            
                            // 记录这个block-query对
                            if (temp_pair_count < max_pairs) {
                                temp_block_ids[temp_pair_count] = block_id;
                                temp_query_ids[temp_pair_count] = qi;
                                temp_pair_count++;
                            }
                        }
                    } else {
                        int block_id = cluster_id;  // 非balanced模式，cluster_id就是block_id
                        
                        // 记录block信息（如果第一次遇到）
                        if (!block_seen[block_id]) {
                            block_seen[block_id] = 1;
                            float* block_ptr = cluster_data[cluster_id][0];
                            if (!block_ptr) {
                                throw std::runtime_error("cluster data pointer is null");
                            }
                            block_host_ptrs[block_id] = block_ptr;
                            block_vec_counts[block_id] = cluster_size[cluster_id];
                            block_global_ids[block_id] = cluster_id;
                        }
                        
                        // 记录这个block-query对
                        if (temp_pair_count < max_pairs) {
                            temp_block_ids[temp_pair_count] = block_id;
                            temp_query_ids[temp_pair_count] = qi;
                            temp_pair_count++;
                        }
                    }
                }
            }
            
            // 第二步：统计活跃block数量，构建compact映射
            active_block_count = 0;
            int* block_id_to_compact = (int*)malloc(max_blocks * sizeof(int));
            for (int i = 0; i < max_blocks; i++) {
                block_id_to_compact[i] = -1;  // -1表示未使用
            }
            
            for (int i = 0; i < max_blocks; i++) {
                if (block_seen[i]) {
                    block_id_to_compact[i] = active_block_count;
                    active_block_count++;
                }
            }
            
            // 分配compact数组
            compact_block_host_ptrs = (float**)malloc(active_block_count * sizeof(float*));
            compact_block_sizes = (int*)malloc(active_block_count * sizeof(int));
            compact_block_ids = (int*)malloc(active_block_count * sizeof(int));
            
            int compact_idx = 0;
            for (int i = 0; i < max_blocks; i++) {
                if (block_seen[i]) {
                    compact_block_host_ptrs[compact_idx] = block_host_ptrs[i];
                    compact_block_sizes[compact_idx] = block_vec_counts[i];
                    compact_block_ids[compact_idx] = block_global_ids[i];
                    compact_idx++;
                }
            }
            
            // 第三步：统计每个block（compact）有多少个query使用它
            int* block_query_count = (int*)calloc(active_block_count, sizeof(int));
            for (int i = 0; i < temp_pair_count; i++) {
                int block_id = temp_block_ids[i];
                int compact_idx_local = block_id_to_compact[block_id];
                if (compact_idx_local >= 0 && compact_idx_local < active_block_count) {
                    block_query_count[compact_idx_local]++;
                }
            }
            
            // 第四步：构建CSR格式的offsets数组
            block_query_offset = (int*)malloc((active_block_count + 1) * sizeof(int));
            block_query_offset[0] = 0;
            for (int i = 0; i < active_block_count; i++) {
                block_query_offset[i + 1] = block_query_offset[i] + block_query_count[i];
            }
            
            // 第五步：分配输出数组
            int total_entries = block_query_offset[active_block_count];
            block_query_data = (int*)malloc(total_entries * sizeof(int));
            block_query_probe_indices = (int*)malloc(total_entries * sizeof(int));
            
            // 第六步：使用临时数组记录每个block的写入位置
            int* block_write_pos = (int*)malloc(active_block_count * sizeof(int));
            for (int i = 0; i < active_block_count; i++) {
                block_write_pos[i] = block_query_offset[i];
            }
            
            // 第七步：为每个query构建probe索引映射（基于compact_idx）
            // 对于每个query，记录它遇到的不同block（compact_idx）及其在query中的索引
            int* query_probe_compact_idx = (int*)malloc(n_query * n_probes * sizeof(int));  // 每个query的probe对应的compact_idx
            int* query_probe_count = (int*)calloc(n_query, sizeof(int));  // 每个query当前有多少个不同的probe
            
            // 重新遍历query-cluster对，构建probe索引映射
            for (int qi = 0; qi < n_query; ++qi) {
                for (int rank = 0; rank < n_probes; ++rank) {
                    int cluster_id = coarse_indices[qi][rank];
                    if (cluster_id < 0 || cluster_id >= n_total_cluster) continue;
                    
                    if (use_balanced_blocks) {
                        int start = cluster2block_offset[cluster_id];
                        int end = cluster2block_offset[cluster_id + 1];
                        for (int idx = start; idx < end; ++idx) {
                            int block_id = cluster2block_ids[idx];
                            if (block_id < 0 || block_id >= n_blocks) continue;
                            
                            int compact_idx_local = block_id_to_compact[block_id];
                            if (compact_idx_local < 0 || compact_idx_local >= active_block_count) continue;
                            
                            // 检查这个compact_idx是否已经在query的probe列表中
                            int probe_index_in_query = -1;
                            int current_probe_count = query_probe_count[qi];
                            for (int p = 0; p < current_probe_count; p++) {
                                if (query_probe_compact_idx[qi * n_probes + p] == compact_idx_local) {
                                    probe_index_in_query = p;
                                    break;
                                }
                            }
                            
                            // 如果没找到，添加新的probe
                            if (probe_index_in_query < 0 && current_probe_count < n_probes) {
                                probe_index_in_query = current_probe_count;
                                query_probe_compact_idx[qi * n_probes + probe_index_in_query] = compact_idx_local;
                                query_probe_count[qi]++;
                            }
                        }
                    } else {
                        int block_id = cluster_id;
                        int compact_idx_local = block_id_to_compact[block_id];
                        if (compact_idx_local < 0 || compact_idx_local >= active_block_count) continue;
                        
                        // 检查这个compact_idx是否已经在query的probe列表中
                        int probe_index_in_query = -1;
                        int current_probe_count = query_probe_count[qi];
                        for (int p = 0; p < current_probe_count; p++) {
                            if (query_probe_compact_idx[qi * n_probes + p] == compact_idx_local) {
                                probe_index_in_query = p;
                                break;
                            }
                        }
                        
                        // 如果没找到，添加新的probe
                        if (probe_index_in_query < 0 && current_probe_count < n_probes) {
                            probe_index_in_query = current_probe_count;
                            query_probe_compact_idx[qi * n_probes + probe_index_in_query] = compact_idx_local;
                            query_probe_count[qi]++;
                        }
                    }
                }
            }
            
            // 第八步：遍历所有query-block对，填充数据（使用正确的probe_index）
            for (int i = 0; i < temp_pair_count; i++) {
                int block_id = temp_block_ids[i];
                int compact_idx_local = block_id_to_compact[block_id];
                if (compact_idx_local >= 0 && compact_idx_local < active_block_count) {
                    int qi = temp_query_ids[i];
                    
                    // 查找这个compact_idx在query中的probe_index
                    int probe_index_in_query = -1;
                    int current_probe_count = query_probe_count[qi];
                    for (int p = 0; p < current_probe_count; p++) {
                        if (query_probe_compact_idx[qi * n_probes + p] == compact_idx_local) {
                            probe_index_in_query = p;
                            break;
                        }
                    }
                    
                    if (probe_index_in_query >= 0) {
                        int write_pos = block_write_pos[compact_idx_local];
                        block_query_data[write_pos] = qi;
                        block_query_probe_indices[write_pos] = probe_index_in_query;
                        block_write_pos[compact_idx_local]++;
                    }
                }
            }
            
            // 清理临时数组
            free(block_seen);
            free(block_host_ptrs);
            free(block_vec_counts);
            free(block_global_ids);
            free(temp_block_ids);
            free(temp_query_ids);
            free(query_probe_compact_idx);
            free(query_probe_count);
            free(block_id_to_compact);
            free(block_query_count);
            free(block_write_pos);
            
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
                        int start = block_query_offset[block_idx];
                        int end = block_query_offset[block_idx + 1];
                        for (int j = start; j < end; j++) {
                            if (block_query_data[j] == qi) {
                                found_blocks.push_back(compact_block_ids[block_idx]);
                                break;  // 找到就退出，避免重复
                            }
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

                    compact_block_host_ptrs,
                    compact_block_sizes,
                    block_query_offset,
                    block_query_data,
                    block_query_probe_indices,  // probe在query中的索引

                    topk_index[0],
                    topk_dist[0],

                    n_query,
                    active_block_count,
                    n_dim,
                    k
                );
                CHECK_CUDA_ERRORS
            }
            
            // 释放Step 2中分配的内存
            if (compact_block_host_ptrs) {
                free(compact_block_host_ptrs);
                compact_block_host_ptrs = nullptr;
            }
            if (compact_block_sizes) {
                free(compact_block_sizes);
                compact_block_sizes = nullptr;
            }
            if (compact_block_ids) {
                free(compact_block_ids);
                compact_block_ids = nullptr;
            }
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

void run_ivf_search_pipeline() {
    // TODO: 后续补充粗筛 + 精筛整体调度
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS
}
