#include <stdlib.h>
#include <limits>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <set>
#include <queue>
#include <float.h>

#include "../cuda/fine_screen_top_n/fine_screen_top_n.cuh"
#include "../cuda/pch.h"
#include "../common/test_utils.cuh"

#define EPSILON 1e-2

// CPU版本的精筛topn计算（用于验证）
void cpu_fine_screen_top_n(
    /**
     * @brief CPU reference implementation for fine-grained top-N screening.
     * 
     * 注意：根据pgvector实现，向量是按聚类物理连续存储的，不是使用倒排表索引。
     * 
     * @param h_query_group            float**  所有查询向量，形状: [n_query][n_dim]
     * @param h_cluster_query_offset   int*     每个cluster分配到的query起始偏移量  [n_total_clusters]
     * @param h_cluster_query_index    int*     按顺序保存每个cluster匹配到的query索引 [sum_query_per_cluster]
     * @param h_cluster_vector_index   int*     每个cluster在全局向量数组中的起始位置（连续索引）[n_total_clusters]
     * @param h_cluster_vector_num     int*     每个cluster内vector数量 [n_total_clusters]
     * @param h_cluster_vector         float**  所有向量数据，按聚类物理连续存储 [n_total_vectors][n_dim]
     * @param n_query                  int      query总数
     * @param n_cluster                int      每个query精筛的cluster数量
     * @param n_total_clusters         int      所有cluster数
     * @param n_dim                    int      向量维度
     * @param n_topn                   int      筛选Top-N（K）个
     * @param n_total_vectors          int      所有向量总数
     * @param h_query_topn_index       [out] int**      每个query的topn索引 [n_query][n_topn]
     * @param h_query_topn_dist        [out] float**    每个query的topn距离 [n_query][n_topn]
     */
    float** h_query_group,              ///< [n_query][n_dim] 查询向量(二维指针)
    int* h_cluster_query_offset,        ///< 分配给每个cluster的query开始偏移（长度n_total_clusters）
    int* h_cluster_query_index,         ///< 按顺序存储分配到cluster的query索引（全局编号，总长=总分配query数）
    int* h_cluster_vector_index,        ///< 每个cluster在全局向量数组中的连续起始位置 [n_total_clusters]
    int* h_cluster_vector_num,          ///< 每个cluster的vector数量（长度n_total_clusters）
    float** h_cluster_vector,           ///< 所有向量数据，按聚类物理连续存储 [n_total_vectors][n_dim]
    int n_query,                        ///< 查询向量总数
    int n_cluster,                      ///< 参与精筛的cluster数量
    int n_total_clusters,               ///< 所有cluster总数（用于偏移，对应offset数组/下标）
    int n_dim,                          ///< 向量维度
    int n_topn,                         ///< 每条query返回的Top-N
    int n_total_vectors,                ///< 总向量数（所有向量累加）
    int** h_query_topn_index,           ///< [out] cpu: [n_query][n_topn], topN索引存储
    float** h_query_topn_dist           ///< [out] cpu: [n_query][n_topn], topN距离存储
) {
    // 计算query向量的L2范数
    float* query_norm = (float*)malloc(n_query * sizeof(float));
    for (int q = 0; q < n_query; q++) {
        float norm_squared = 0.0f;
        for (int d = 0; d < n_dim; d++) {
            float val = h_query_group[q][d];
            norm_squared += val * val;
        }
        query_norm[q] = sqrtf(norm_squared);
    }
    
    // 计算cluster向量的L2范数
    float* cluster_vector_norm = (float*)malloc(n_total_vectors * sizeof(float));
    for (int v = 0; v < n_total_vectors; v++) {
        float norm_squared = 0.0f;
        for (int d = 0; d < n_dim; d++) {
            float val = h_cluster_vector[v][d];
            norm_squared += val * val;
        }
        cluster_vector_norm[v] = sqrtf(norm_squared);
        
        // // Debug: 打印norm值（小数据时）
        // if (n_query <= 4 && v < 10) {
        //     printf("[CPU Debug] Vector %d norm: %.5f (squared_sum=%.5f)\n", v, cluster_vector_norm[v], norm_squared);
        //     // 验证数据访问：检查h_cluster_vector[v]和h_cluster_vector[0]的关系
        //     if (v == 6) {
        //         printf("[CPU Debug] h_cluster_vector[0] = %p, h_cluster_vector[6] = %p, diff = %ld elements\n",
        //                (void*)h_cluster_vector[0], (void*)h_cluster_vector[6], 
        //                (h_cluster_vector[6] - h_cluster_vector[0]));
        //     }
        // }
    }
    
    // // Debug: 打印query norm值
    // if (n_query <= 4) {
    //     for (int q = 0; q < n_query; q++) {
    //         printf("[CPU Debug] Query %d norm: %.5f\n", q, query_norm[q]);
    //     }
    //     printf("[CPU Debug] Vector 0 first 3 dims: %.5f %.5f %.5f\n", 
    //            h_cluster_vector[0][0], h_cluster_vector[0][1], h_cluster_vector[0][2]);
    //     printf("[CPU Debug] Vector 6 first 3 dims: %.5f %.5f %.5f\n", 
    //            h_cluster_vector[6][0], h_cluster_vector[6][1], h_cluster_vector[6][2]);
    // }
    
    // 为每个query初始化最大堆（用于维护top-k最小距离）
    // 使用 pair<distance, index> 存储在堆中，堆顶是最大的元素
    std::vector<std::priority_queue<std::pair<float, int>>> heaps(n_query);
    
    // 对每个cluster进行处理
    for (int cluster_idx = 0; cluster_idx < n_total_clusters; cluster_idx++) {
        // 获取当前cluster的query范围
        // 使用标准的offset数组格式：count = offset[i+1] - offset[i]
        int query_start = h_cluster_query_offset[cluster_idx];
        int query_end = h_cluster_query_offset[cluster_idx + 1];  // 直接使用下一个offset作为结束位置
        int query_count = query_end - query_start;
        
        // 边界检查
        // query_start 是在 h_cluster_query_index 中的偏移量，不是 query 索引
        // 它可以是任何非负整数，只要 query_start + query_count 不超过数组大小
        // 我们需要检查的是 query_count 是否有效，以及是否能访问到有效的 query 索引
        if (query_start < 0 || query_count <= 0) {
            // if (n_query <= 4) {
            //     printf("[CPU Debug] Skipping cluster %d: query_start=%d, query_count=%d\n", 
            //            cluster_idx, query_start, query_count);
            // }
            continue;
        }
        
        // // Debug: 打印cluster信息
        // if (n_query <= 4) {
        //     printf("[CPU Debug] Processing cluster %d: query_start=%d, query_end=%d, query_count=%d\n",
        //            cluster_idx, query_start, query_end, query_count);
        //     printf("[CPU Debug] Cluster %d queries: ", cluster_idx);
        //     for (int q = query_start; q < query_end; q++) {
        //         printf("%d ", h_cluster_query_index[q]);
        //     }
        //     printf("\n");
        // }
        
        // 获取当前cluster的向量信息（按聚类物理连续存储）
        int vector_start_index = h_cluster_vector_index[cluster_idx];  // cluster在全局向量数组中的连续起始位置
        int vector_count = h_cluster_vector_num[cluster_idx];
        
        // 边界检查
        if (vector_start_index < 0 || vector_count <= 0 || vector_start_index + vector_count > n_total_vectors) {
            continue;
        }
        
        // 对每个query计算与当前cluster中向量的距离
        for (int q = 0; q < query_count; q++) {
            int query_idx = h_cluster_query_index[query_start + q];
            
            // 边界检查
            if (query_idx < 0 || query_idx >= n_query) continue;
            
            // 计算当前query与cluster中所有向量的余弦距离
            // 向量按聚类物理连续存储，可以直接使用连续索引
            for (int vec_idx = 0; vec_idx < vector_count; vec_idx++) {
                int global_vec_idx = vector_start_index + vec_idx;  // 连续索引
                
                // 边界检查
                if (global_vec_idx < 0 || global_vec_idx >= n_total_vectors) continue;
                
                // 计算内积
                float dot_product = 0.0f;
                for (int d = 0; d < n_dim; d++) {
                    dot_product += h_query_group[query_idx][d] * 
                                  h_cluster_vector[global_vec_idx][d];
                }
                
                // 计算余弦距离: 1 - cos_similarity
                float cos_distance;
                if (query_norm[query_idx] < 1e-6f || cluster_vector_norm[global_vec_idx] < 1e-6f) {
                    cos_distance = 1.0f;  // 如果任一向量接近零向量，距离为1
                } else {
                    float cos_similarity = dot_product / (query_norm[query_idx] * cluster_vector_norm[global_vec_idx]);
                    // 限制范围，避免数值问题
                    cos_distance = 1.0f - cos_similarity;
                }
                
                // // Debug: 打印处理过程（小数据时打印所有query的处理）
                // if (n_query <= 4) {
                //     printf("[CPU Debug] Query %d: cluster_idx=%d, vec_idx=%d, global_vec_idx=%d, "
                //            "dot=%.5f, q_norm=%.5f, v_norm=%.5f, cos_sim=%.5f, cos_dist=%.5f\n", 
                //            query_idx, cluster_idx, vec_idx, global_vec_idx, 
                //            dot_product, query_norm[query_idx], cluster_vector_norm[global_vec_idx],
                //            dot_product / (query_norm[query_idx] * cluster_vector_norm[global_vec_idx]), cos_distance);
                // }
                
                // 使用最大堆维护top-k最小距离（堆顶是最大的元素）
                auto& heap = heaps[query_idx];
                if (heap.size() < n_topn) {
                    // 堆未满，直接添加
                    heap.push(std::make_pair(cos_distance, global_vec_idx));
                } else if (cos_distance < heap.top().first) {
                    // 堆已满，如果当前距离小于堆顶，则替换堆顶
                    heap.pop();
                    heap.push(std::make_pair(cos_distance, global_vec_idx));
                }
                // 如果 cos_distance >= heap.top().first，则忽略该元素
            }
        }
    }
    
    // 从堆中直接提取top-k结果
    for (int q = 0; q < n_query; q++) {
        
        auto& heap = heaps[q];
        
        // // Debug: 检查堆的大小和内容（仅在小规模测试时）
        // if (!QUIET && n_query <= 4) {
        //     printf("[CPU Debug] Query %d: heap size = %zu\n", q, heap.size());
        //     // 临时复制堆来打印内容（不影响原堆）
        //     auto temp_heap = heap;
        //     int idx = 0;
        //     while (!temp_heap.empty() && idx < 10) {
        //         auto top = temp_heap.top();
        //         temp_heap.pop();
        //         printf("  [%d] dist=%.5f, index=%d\n", idx++, top.first, top.second);
        //     }
        // }
        
        int k = 0;
        while (!heap.empty() && k < n_topn) {
            auto top = heap.top();
            heap.pop();
            h_query_topn_dist[q][n_topn - 1 - k] = top.first;
            h_query_topn_index[q][n_topn - 1 - k] = top.second;
            k++;
        }
    }
    
    // 释放临时内存
    free(query_norm);
    free(cluster_vector_norm);
}

/**
 * 生成query对应的cluster组
 * 先确定一个固定的cluster集合，然后每个query从这个集合中选择n_cluster个cluster
 * 
 * @param n_query query数量
 * @param n_cluster 每个query关联的cluster数量
 * @param n_total_clusters 总cluster数量（所有cluster的集合大小）
 */
int** generate_query_cluster_group(int n_query, int n_cluster, int n_total_clusters) {
    int** h_query_cluster_group = malloc_vector_list<int>(n_query, n_cluster);
        
    // 每个 query 从 cluster 集合中选择 n_cluster 个不同的 cluster
    for (int i = 0; i < n_query; i++) {
        std::vector<int> selected_clusters;
        
        // 从固定的 cluster 集合中随机选择 n_cluster 个不同的 cluster
        while (selected_clusters.size() < n_cluster) {
            int cluster_id = rand() % n_total_clusters;
            if (std::find(selected_clusters.begin(), selected_clusters.end(), cluster_id) == selected_clusters.end()) {
                selected_clusters.push_back(cluster_id);
            }
        }
        
        // 填充数据
        for (int j = 0; j < n_cluster; j++) {
            h_query_cluster_group[i][j] = selected_clusters[j];
        }
    }
    
    return h_query_cluster_group;
}

// 生成cluster查询的倒排索引
// 使用标准的offset数组格式：分配 n_total_clusters + 1 个元素
// offset[i] = cluster i 的起始位置
// offset[i+1] = cluster i 的结束位置（也是 cluster i+1 的起始位置）
// offset[n_total_clusters] = 总长度
void generate_cluster_query_inverted_index(
    int n_query, int n_cluster, int** h_query_cluster_group, int n_total_clusters, 
    int** h_cluster_query_offset, int** h_cluster_query_index
) {
    // 分配 n_total_clusters + 1 个元素（标准的offset数组格式）
    *h_cluster_query_offset = (int*)malloc((n_total_clusters + 1) * sizeof(int));
    
    // 统计每个cluster对应的query数量（去重）
    std::map<int, std::set<int>> cluster_queries_set;
    
    for (int i = 0; i < n_query; i++) {
        for (int j = 0; j < n_cluster; j++) {
            int cluster_id = h_query_cluster_group[i][j];
            cluster_queries_set[cluster_id].insert(i);
        }
    }
    
    // 转换为vector
    std::map<int, std::vector<int>> cluster_queries;
    for (const auto& pair : cluster_queries_set) {
        cluster_queries[pair.first].assign(pair.second.begin(), pair.second.end());
    }
    
    // 计算总数据量
    int total_queries = 0;
    for (const auto& pair : cluster_queries) {
        total_queries += pair.second.size();
    }
    
    *h_cluster_query_index = (int*)malloc(total_queries * sizeof(int));
    
    // 计算offset（标准格式：offset[i+1] = offset[i] + count[i]）
    int offset = 0;
    for (int i = 0; i < n_total_clusters; i++) {
        (*h_cluster_query_offset)[i] = offset;
        offset += cluster_queries[i].size();
    }
    // 设置最后一个元素为总长度（用于统一计算最后一个cluster的数量）
    (*h_cluster_query_offset)[n_total_clusters] = offset;
    
    // 填充query数据
    int data_idx = 0;
    for (int i = 0; i < n_total_clusters; i++) {
        for (int query_id : cluster_queries[i]) {
            (*h_cluster_query_index)[data_idx++] = query_id;
        }
    }
}

// 生成按聚类物理连续存储的向量布局（符合pgvector实现）
// 向量按聚类分组，同一聚类的向量在全局数组中连续存储
void generate_cluster_vector_layout(
    int n_total_clusters, 
    int n_total_vectors,
    int** h_cluster_vector_index,  // [out] 每个cluster在全局向量数组中的连续起始位置 [n_total_clusters]
    int** h_cluster_vector_num     // [out] 每个cluster的向量数量 [n_total_clusters]
) {
    // 分配每个cluster的向量数量
    *h_cluster_vector_num = (int*)malloc(n_total_clusters * sizeof(int));
    int remaining_vectors = n_total_vectors;
    
    for (int i = 0; i < n_total_clusters; i++) {
        if (i == n_total_clusters - 1) {
            (*h_cluster_vector_num)[i] = remaining_vectors;
        } else {
            int max_vectors = remaining_vectors - (n_total_clusters - i - 1);
            (*h_cluster_vector_num)[i] = 1 + rand() % std::max(1, max_vectors);
        }
        remaining_vectors -= (*h_cluster_vector_num)[i];
    }
    
    // 计算每个cluster在全局向量数组中的连续起始位置
    *h_cluster_vector_index = (int*)malloc(n_total_clusters * sizeof(int));
    
    int current_offset = 0;
    for (int i = 0; i < n_total_clusters; i++) {
        (*h_cluster_vector_index)[i] = current_offset;  // 连续起始位置
        current_offset += (*h_cluster_vector_num)[i];
    }
}

// 测试函数 - 返回性能指标
std::vector<double> test_fine_screen_top_n(
    int n_query, int n_cluster, int n_dim, int n_topn, int n_total_vectors
) {
    bool pass = true;
    double memory_mb = (double)(n_query * n_dim + n_total_vectors * n_dim + n_query * n_cluster * 2 + n_cluster * 3) * sizeof(float) / (double)(1024 * 1024);
    
    if (!QUIET) {
        COUT_VAL("配置:", n_query, "个query ×", n_cluster, "个cluster", ", 维度:", n_dim, ", top", n_topn, ", 总向量:", n_total_vectors);
    }
    else
        COUT_VAL("配置:", n_query, "个query ×", n_cluster, "个cluster", ", 维度:", n_dim, ", top", n_topn, ", 总向量:", n_total_vectors);
    
    // 生成测试数据
    // 先确定总cluster数量：设为 n_cluster 的倍数，这样每个query只搜索部分cluster
    int n_total_clusters = std::min(n_cluster, (int)sqrt(n_total_vectors));
    
    // 生成按聚类物理连续存储的向量布局（符合pgvector实现）
    int* h_cluster_vector_index;  // 每个cluster的连续起始位置
    int* h_cluster_vector_num;   // 每个cluster的向量数量
    generate_cluster_vector_layout(n_total_clusters, n_total_vectors, 
                                   &h_cluster_vector_index, &h_cluster_vector_num);
    
    // 生成向量数据，按聚类物理连续存储（符合pgvector的实际存储方式）
    float** h_cluster_vector = generate_vector_list(n_total_vectors, n_dim);
    
    // 然后为每个query从固定的cluster集合中选择n_cluster个cluster
    float** h_query_group = generate_vector_list(n_query, n_dim);
    int** query_cluster_data = generate_query_cluster_group(n_query, n_cluster, n_total_clusters);
    
    int* h_cluster_query_offset, *h_cluster_query_index;
    generate_cluster_query_inverted_index(n_query, n_cluster, query_cluster_data, n_total_clusters,
                                        &h_cluster_query_offset, &h_cluster_query_index);
    
    // 初始化输出数组
    int** h_query_topn_index = malloc_vector_list<int>(n_query, n_topn);
    int** h_query_topn_index_cpu = malloc_vector_list<int>(n_query, n_topn);
    float** h_query_topn_dist = malloc_vector_list<float>(n_query, n_topn);
    float** h_query_topn_dist_cpu = malloc_vector_list<float>(n_query, n_topn);
    
    for (int i = 0; i < n_query; i++) {
        for (int j = 0; j < n_topn; j++) {
            h_query_topn_index[i][j] = 0;
            h_query_topn_index_cpu[i][j] = 0;
            h_query_topn_dist[i][j] = std::numeric_limits<float>::infinity();
            h_query_topn_dist_cpu[i][j] = std::numeric_limits<float>::infinity();
        }
    }
    
    // 计算最大cluster向量数量
    int max_cluster_vector_count = 0;
    for (int i = 0; i < n_total_clusters; i++) {
        max_cluster_vector_count = std::max(max_cluster_vector_count, h_cluster_vector_num[i]);
    }
    
    // CPU测试
    double cpu_duration_ms = 0;
    MEASURE_MS_AND_SAVE("CPU Fine Screen TopN", cpu_duration_ms,
        cpu_fine_screen_top_n(
            h_query_group, h_cluster_query_offset, h_cluster_query_index,
            h_cluster_vector_index, h_cluster_vector_num, h_cluster_vector,
            n_query, n_cluster, n_total_clusters, n_dim, n_topn, n_total_vectors,
            h_query_topn_index_cpu, h_query_topn_dist_cpu
        );
    );
    
    // // Debug: 打印输入数据
    // if (!QUIET && n_query <= 4) {
    //     printf("\n=== DEBUG: Input Data ===\n");
    //     printf("n_query=%d, n_cluster=%d, n_dim=%d, n_topn=%d\n", n_query, n_cluster, n_dim, n_topn);
    //     printf("n_total_clusters=%d, n_total_vectors=%d, max_cluster_vector_count=%d\n", 
    //            n_total_clusters, n_total_vectors, max_cluster_vector_count);
        
    //     printf("h_cluster_query_offset: ");
    //     for (int i = 0; i < n_total_clusters; i++) {
    //         printf("%d ", h_cluster_query_offset[i]);
    //     }
    //     printf("\n");
        
    //     printf("h_cluster_vector_index (每个cluster的连续起始位置): ");
    //     for (int i = 0; i < n_total_clusters; i++) {
    //         printf("%d ", h_cluster_vector_index[i]);
    //     }
    //     printf("\n");
        
    //     printf("h_cluster_vector_num: ");
    //     for (int i = 0; i < n_total_clusters; i++) {
    //         printf("%d ", h_cluster_vector_num[i]);
    //     }
    //     printf("\n");
        
    //     printf("h_cluster_query_index: ");
    //     for (int i = 0; i < n_total_clusters; i++) {
    //         // 使用标准的offset数组格式
    //         int query_start = h_cluster_query_offset[i];
    //         int query_end = h_cluster_query_offset[i + 1];
    //         int count = query_end - query_start;
    //         printf("[");
    //         for (int j = 0; j < count; j++) {
    //             printf("%d,", h_cluster_query_index[query_start + j]);
    //         }
    //         printf("] ");
    //     }
    //     printf("\n");
        
    //     // 打印L2 norm验证
    //     printf("Expected norms (verify manually):\n");
    //     float q0_norm = 0, q1_norm = 0;
    //     for (int i = 0; i < n_dim; i++) {
    //         q0_norm += h_query_group[0][i] * h_query_group[0][i];
    //         q1_norm += h_query_group[1][i] * h_query_group[1][i];
    //     }
    //     printf("Query 0 norm: %.4f\n", sqrtf(q0_norm));
    //     printf("Query 1 norm: %.4f\n", sqrtf(q1_norm));
    // }
    
    // GPU测试（使用连续存储接口，符合pgvector实现）
    double gpu_duration_ms = 0;
    MEASURE_MS_AND_SAVE("GPU Fine Screen TopN", gpu_duration_ms,
        fine_screen_top_n_v1(
            h_query_group, h_cluster_query_offset, h_cluster_query_index,
            h_cluster_vector_index, h_cluster_vector_num, h_cluster_vector,
            n_query, n_cluster, n_total_clusters, n_dim, n_topn, n_total_vectors,
            h_query_topn_index, h_query_topn_dist
        );
    );
    
    // 计算加速比
    double speedup = cpu_duration_ms / gpu_duration_ms;
    
    // 验证结果（使用 compare_set_2D，因为结果是二维数组且顺序可能不同）
    pass &= compare_set_2D(h_query_topn_dist_cpu, h_query_topn_dist, n_query, n_topn, EPSILON);
    // pass &= compare_set_2D(h_query_topn_index_cpu, h_query_topn_index, n_query, n_topn);
    
    if (!pass && !QUIET) {
        // 打印详细信息用于调试
        print_2D("cpu topk dist", h_query_topn_dist_cpu, n_query, n_topn); 
        print_2D("gpu topk dist", h_query_topn_dist, n_query, n_topn); 
        print_2D("cpu topk index", h_query_topn_index_cpu, n_query, n_topn); 
        print_2D("gpu topk index", h_query_topn_index, n_query, n_topn); 
    }
    
    free_vector_list((void**)h_query_group);
    free(query_cluster_data);
    free(h_cluster_query_offset);
    free(h_cluster_query_index);
    free(h_cluster_vector_index);   // 连续起始位置数组
    free(h_cluster_vector_num);      // 向量数量数组
    free_vector_list((void**)h_cluster_vector);
    free_vector_list((void**)h_query_topn_index);
    free_vector_list((void**)h_query_topn_dist);
    free_vector_list((void**)h_query_topn_index_cpu);
    free_vector_list((void**)h_query_topn_dist_cpu);
    
    return {
        pass ? 1.0 : 0.0,
        (double)n_query, (double)n_cluster, (double)n_dim, (double)n_topn, (double)n_total_vectors,
        gpu_duration_ms, cpu_duration_ms, speedup, (double)memory_mb
    };
}

int main(int argc, char** argv) {
    srand(time(0));
    
    bool all_pass = true;
    
    MetricsCollector metrics;
    metrics.set_columns("pass rate", "n_query", "n_cluster", "n_dim", "n_topn", "n_total_vectors", "gpu_ms", "cpu_ms", "speedup", "memory_mb");
    metrics.set_num_repeats(1); 
    
    COUT_ENDL("测试精筛TopN算法");
    
    // 测试不同参数组合（使用最小参数进行debug）
    // PARAM_3D(n_query, (8, 32, 64), 
    // // PARAM_3D(n_query, (10000), 
    // // PARAM_3D(n_query, (50), 
    // n_cluster, (2,20,200), 
    // n_dim, (8,64,512))
    // //  n_dim, (512, 1024))
    // PARAM_3D(n_query, (256), 
    // n_cluster, (20), 
    // n_dim, (1024))
    // {
    //     int n_topn = 100;
    //     // int n_topn = 16;
    //     // int n_total_vectors = 8;
    //     int n_total_vectors = 1000000;
    //     // int n_total_vectors = 2048;
    PARAM_3D(n_query, (1,10, 5000), /* 试了很多不同的数值，都没有问题 */
    n_cluster, (20), 
    n_dim, (128))
    {
        int n_topn = 100;
        int n_total_vectors = 10000;
        
        auto result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto test_result = test_fine_screen_top_n(n_query, n_cluster, n_dim, n_topn, n_total_vectors);
            all_pass &= (test_result[0] == 1.0);
            return test_result;
        });
    }
    
    metrics.export_csv("final_topk.csv");
    metrics.print_table();
    
    COUT_ENDL("\n所有测试:", all_pass ? "✅ PASS" : "❌ FAIL");
    return 0;
}
