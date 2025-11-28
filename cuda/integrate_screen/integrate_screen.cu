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

                           int n_query,
                           int n_dim,
                           int n_total_cluster,
                           int n_probes,
                           int k) {

    if (n_query <= 0 || n_dim <= 0 || n_total_cluster <= 0 || k <= 0) {
        printf("[ERROR] Invalid parameters: n_query=%d, n_dim=%d, n_total_cluster=%d, k=%d\n",
               n_query, n_dim, n_total_cluster, k);
        throw std::invalid_argument("invalid batch_search_pipeline configuration");
    }
    if (!cluster_size || !cluster_data) {
        throw std::invalid_argument("cluster metadata is null");
    }

    if (!cluster_center_data) {
        throw std::invalid_argument("cluster_center_data must not be null for coarse search");
    }
    if (n_probes <= 0 || n_probes > n_total_cluster) {
        throw std::invalid_argument("invalid n_probes");
    }

    int** coarse_indices = (int**)malloc_vector_list(n_query, n_probes, sizeof(int));
    float** coarse_dists = (float**)malloc_vector_list(n_query, n_probes, sizeof(float));

    float* d_queries = nullptr;
    float** d_clusters = (float**)malloc(n_total_cluster * sizeof(float*));

    cudaMalloc(&d_queries, n_query * n_dim * sizeof(float));
    cudaMemcpy(d_queries, query_batch[0], n_query * n_dim * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS

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

    // ------------------------------------------------------------------
    // Step 2. 将 query→cluster 粗筛结果转成 cluster 序列
    // ------------------------------------------------------------------
    int* cluster_query_offset = nullptr;
    int* cluster_query_data = nullptr;
    int* cluster_query_probe_indices = nullptr;  // 每个cluster-query对中probe在query中的索引

    {
        CUDATimer timer("Step 2: Convert query→cluster to cluster sequence", true, false);
        
        // 第一步：统计每个cluster有多少个query使用它
        int* cluster_query_count = (int*)calloc(n_total_cluster, sizeof(int));
        for (int qi = 0; qi < n_query; ++qi) {
            for (int rank = 0; rank < n_probes; ++rank) {
                int cluster_id = coarse_indices[qi][rank];
                cluster_query_count[cluster_id]++;
            }
        }
        
        // 第二步：构建CSR格式的offsets数组
        cluster_query_offset = (int*)malloc((n_total_cluster + 1) * sizeof(int));
        cluster_query_offset[0] = 0;
        for (int i = 0; i < n_total_cluster; i++) {
            cluster_query_offset[i + 1] = cluster_query_offset[i] + cluster_query_count[i];
        }
        
        // 第三步：分配输出数组
        int total_entries = cluster_query_offset[n_total_cluster];
        cluster_query_data = (int*)malloc(total_entries * sizeof(int));
        cluster_query_probe_indices = (int*)malloc(total_entries * sizeof(int));
        
        // 第四步：使用临时数组记录每个cluster的写入位置
        int* cluster_write_pos = (int*)malloc(n_total_cluster * sizeof(int));
        for (int i = 0; i < n_total_cluster; i++) {
            cluster_write_pos[i] = cluster_query_offset[i];
        }
        
        // 第五步：遍历所有query-cluster对，填充数据
        // 注意：粗筛结果中每个query的cluster不重复，所以rank就是probe_index_in_query
        for (int qi = 0; qi < n_query; ++qi) {
            for (int rank = 0; rank < n_probes; ++rank) {
                int cluster_id = coarse_indices[qi][rank];
                int write_pos = cluster_write_pos[cluster_id];
                cluster_query_data[write_pos] = qi;
                cluster_query_probe_indices[write_pos] = rank;  // rank就是probe_index_in_query
                cluster_write_pos[cluster_id]++;
            }
        }
        
        // 清理临时数组
        free(cluster_query_count);
        free(cluster_write_pos);
        
        // 输出 Step 2 结果，验证是否由粗筛结果转换而来
        // if (n_query <= 4 && n_probes <= 8) {
        if (false) {
            bool all_match = true;
            for (int qi = 0; qi < n_query; ++qi) {
                // printf("Query %d: ", qi);
                std::vector<int> expected_blocks;
                for (int rank = 0; rank < n_probes; ++rank) {
                    int cluster_id = coarse_indices[qi][rank];
                    if (cluster_id < 0 || cluster_id >= n_total_cluster) continue;
                    
                    expected_blocks.push_back(cluster_id);
                }
                
                // // 检查这些 block 是否都在 Step 2 的输出中，并且包含 query qi
                // printf("expected blocks: [");
                // for (size_t i = 0; i < expected_blocks.size(); ++i) {
                //     printf("%d", expected_blocks[i]);
                //     if (i < expected_blocks.size() - 1) printf(", ");
                // }
                // printf("] -> ");
                
                std::vector<int> found_clusters;
                for (int cluster_id = 0; cluster_id < n_total_cluster; ++cluster_id) {
                    int start = cluster_query_offset[cluster_id];
                    int end = cluster_query_offset[cluster_id + 1];
                    for (int j = start; j < end; j++) {
                        if (cluster_query_data[j] == qi) {
                            found_clusters.push_back(cluster_id);
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
                if (expected_blocks.size() != found_clusters.size()) {
                    all_match = false;
                } else {
                    std::sort(expected_blocks.begin(), expected_blocks.end());
                    std::sort(found_clusters.begin(), found_clusters.end());
                    all_match &= (expected_blocks == found_clusters);
                }
            }
            printf("step 2 all_match: %s\n", all_match ? "true!!!" : "false");
        }
    }

    COUT_ENDL("n_query: ", n_query, "n_probes: ", n_probes, "k: ", k);

    // ------------------------------------------------------------------
    // Step 3. 精筛：上传 cluster 向量 + cluster 查询映射，调用 GPU kernel
    // ------------------------------------------------------------------
    COUT_ENDL("Step 3: Fine Search (fine_screen_top_n_blocks)");
    {
        CUDATimer timer("Step 3: Fine Search (fine_screen_top_n_blocks)");
        // 构建cluster_vectors数组（float**），每个元素指向对应cluster的数据
        float** cluster_vectors = (float**)malloc(n_total_cluster * sizeof(float*));
        for (int i = 0; i < n_total_cluster; ++i) {
            cluster_vectors[i] = cluster_data[i][0];
        }
        
        // 调用固定probe版本的精筛，直接使用完整的cluster数组
        fine_screen_topk(
            query_batch[0],

            cluster_vectors,  // float**，每个元素指向对应cluster的数据
            cluster_size,     // int*，每个cluster的向量数量
            cluster_query_offset,  // 大小为(n_total_cluster + 1)，包含所有cluster
            cluster_query_data,
            cluster_query_probe_indices,  // probe在query中的索引

            topk_index,
            topk_dist,

            n_query,
            n_total_cluster,  // 总的cluster数量
            n_probes,  // 每个query的probe数量
            n_dim,
            k
        );
        CHECK_CUDA_ERRORS
        
        free(cluster_vectors);
    }
    
    // 释放Step 2中分配的内存
    if (cluster_query_offset) {
        free(cluster_query_offset);
        cluster_query_offset = nullptr;
    }
    if (cluster_query_data) {
        free(cluster_query_data);
        cluster_query_data = nullptr;
    }
    if (cluster_query_probe_indices) {
        free(cluster_query_probe_indices);
        cluster_query_probe_indices = nullptr;
    }

    cudaFree(d_queries);

    for (int i = 0; i < n_total_cluster; ++i) {
        if (d_clusters[i]) {
            cudaFree(d_clusters[i]);
            d_clusters[i] = nullptr;
        }
    }
    free(d_clusters);
    // printf("after free\n");
    
    free_vector_list((void**)coarse_indices);
    free_vector_list((void**)coarse_dists);
}

void run_integrate_pipeline() {
    // TODO: 后续补充粗筛 + 精筛整体调度
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS
}
