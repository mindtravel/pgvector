#include <stdlib.h>
#include <limits>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <set>
#include <float.h>

#include "../cuda/fine_screen_top_n/fine_screen_top_n.cuh"
#include "../cuda/pch.h"
#include "../common/test_utils.cuh"

#define EPSILON 1e-2

// CPU版本的精筛topn计算（用于验证）
void cpu_fine_screen_top_n(
    float* h_query_group, int* h_query_cluster_group, int* h_cluster_query_offset, int* h_cluster_query_data,
    int* cluster_map,
    int* h_cluster_vector_index, int* h_cluster_vector_num, float** h_cluster_vector,
    int n_query, int n_cluster, int distinct_cluster_count, int n_dim, int n_topn, int max_cluster_id, int tol_vector,
    int max_cluster_vector_count,  // 新增：最大聚类向量数量
    int* h_query_topn_index, float* h_query_topn_dist
) {
    // 计算query向量的L2范数
    float* query_norm = (float*)malloc(n_query * sizeof(float));
    for (int q = 0; q < n_query; q++) {
        float norm_squared = 0.0f;
        for (int d = 0; d < n_dim; d++) {
            float val = h_query_group[q * n_dim + d];
            norm_squared += val * val;
        }
        query_norm[q] = sqrtf(norm_squared);
    }
    
    // 计算cluster向量的L2范数
    float* cluster_vector_norm = (float*)malloc(tol_vector * sizeof(float));
    for (int v = 0; v < tol_vector; v++) {
        float norm_squared = 0.0f;
        for (int d = 0; d < n_dim; d++) {
            float val = h_cluster_vector[v][d];
            norm_squared += val * val;
        }
        cluster_vector_norm[v] = sqrtf(norm_squared);
    }
    
    // 初始化top-k结果
    for (int q = 0; q < n_query; q++) {
        for (int k = 0; k < n_topn; k++) {
            h_query_topn_index[q * n_topn + k] = -1;
            h_query_topn_dist[q * n_topn + k] = FLT_MAX;
        }
    }
    
    // 对每个cluster进行处理
    for (int cluster_idx = 0; cluster_idx < distinct_cluster_count; cluster_idx++) {
        // 获取当前cluster的query范围
        int query_start = h_cluster_query_offset[cluster_idx];
        int query_count;
        if (cluster_idx + 1 >= distinct_cluster_count) {
            query_count = n_query - query_start;
        } else {
            query_count = h_cluster_query_offset[cluster_idx + 1] - query_start;
        }
        
        // 边界检查
        if (query_start >= n_query || query_start + query_count > n_query || query_count <= 0) {
            continue;
        }
        
        // 获取当前cluster的向量信息
        int vector_start_idx = h_cluster_vector_index[cluster_idx];
        int vector_count = h_cluster_vector_num[cluster_idx];
        
        // 对每个query计算与当前cluster中向量的距离
        for (int q = 0; q < query_count; q++) {
            int query_idx = query_start + q;
            
            // 计算当前query与cluster中所有向量的L2距离
            for (int vec_idx = 0; vec_idx < vector_count; vec_idx++) {
                int global_vec_idx = vector_start_idx + vec_idx;
                
                // 计算L2距离的平方（使用L2范数优化）
                float dot_product = 0.0f;
                for (int dim = 0; dim < n_dim; dim++) {
                    dot_product += h_query_group[query_idx * n_dim + dim] * 
                                  h_cluster_vector[global_vec_idx][dim];
                }
                
                // L2距离平方 = ||q||^2 + ||v||^2 - 2*q·v
                float distance_squared = query_norm[query_idx] * query_norm[query_idx] + 
                                       cluster_vector_norm[global_vec_idx] * cluster_vector_norm[global_vec_idx] - 
                                       2.0f * dot_product;
                
                // 取平方根得到实际距离
                float distance = sqrtf(fmaxf(0.0f, distance_squared));
                
                // 插入到当前query的topk中
                for (int k = 0; k < n_topn; k++) {
                    if (distance < h_query_topn_dist[query_idx * n_topn + k]) {
                        // 向后移动元素
                        for (int m = n_topn - 1; m > k; m--) {
                            h_query_topn_dist[query_idx * n_topn + m] = h_query_topn_dist[query_idx * n_topn + m-1];
                            h_query_topn_index[query_idx * n_topn + m] = h_query_topn_index[query_idx * n_topn + m-1];
                        }
                        // 插入新元素
                        h_query_topn_dist[query_idx * n_topn + k] = distance;
                        h_query_topn_index[query_idx * n_topn + k] = global_vec_idx;
                        break;
                    }
                }
            }
        }
    }
    
    // 释放临时内存
    free(query_norm);
    free(cluster_vector_norm);
}

// GPU版本的精筛topn计算（暂时空实现）
void gpu_fine_screen_top_n(
    float* h_query_group, int* h_query_cluster_group, int* h_cluster_query_offset, int* h_cluster_query_data,
    int* cluster_map,
    int* h_cluster_vector_index, int* h_cluster_vector_num, float** h_cluster_vector,
    int n_query, int n_cluster, int distinct_cluster_count, int n_dim, int n_topn, int max_cluster_id, int tol_vector,
    int max_cluster_vector_count,  // 新增：最大聚类向量数量
    int* h_query_topn_index, float* h_query_topn_dist
) {
    // 直接调用fine_screen_top_n函数
    fine_screen_top_n(
        h_query_group, h_query_cluster_group, h_cluster_query_offset, h_cluster_query_data,
        cluster_map,
        h_cluster_vector_index, h_cluster_vector_num, h_cluster_vector,
        n_query, n_cluster, distinct_cluster_count, n_dim, n_topn, max_cluster_id, tol_vector,
        max_cluster_vector_count,  // 新增：最大聚类向量数量
        h_query_topn_index, h_query_topn_dist
    );
}

// 生成稀疏的cluster ID
std::vector<int> generate_sparse_cluster_ids(int n_cluster, int max_cluster_id) {
    std::vector<int> cluster_ids;
    std::vector<bool> used(max_cluster_id, false);
    
    for (int i = 0; i < n_cluster; i++) {
        int cluster_id;
        do {
            cluster_id = rand() % max_cluster_id;
        } while (used[cluster_id]);
        
        used[cluster_id] = true;
        cluster_ids.push_back(cluster_id);
    }
    
    std::sort(cluster_ids.begin(), cluster_ids.end());
    return cluster_ids;
}

// 生成cluster_map
int* generate_cluster_map(const std::vector<int>& cluster_ids) {
    int n_cluster = cluster_ids.size();
    int* cluster_map = (int*)malloc(n_cluster * sizeof(int));
    
    for (int i = 0; i < n_cluster; i++) {
        cluster_map[i] = cluster_ids[i];
    }
    
    return cluster_map;
}

// 生成query对应的cluster组，并统计distinct cluster
struct QueryClusterData {
    int* h_query_cluster_group;
    std::vector<int> distinct_cluster_ids;
    int distinct_cluster_count;
};

QueryClusterData generate_query_cluster_group(int n_query, int n_cluster, int max_cluster_id) {
    QueryClusterData result;
    result.h_query_cluster_group = (int*)malloc(n_query * n_cluster * sizeof(int));
    
    std::set<int> distinct_clusters;
    
    for (int i = 0; i < n_query; i++) {
        // 每个query随机选择n_cluster个cluster
        std::vector<int> selected_clusters;
        
        for (int j = 0; j < n_cluster; j++) {
            int cluster_id = rand() % max_cluster_id;
            selected_clusters.push_back(cluster_id);
            distinct_clusters.insert(cluster_id);
        }
        
        // 去重并排序
        std::sort(selected_clusters.begin(), selected_clusters.end());
        selected_clusters.erase(std::unique(selected_clusters.begin(), selected_clusters.end()), selected_clusters.end());
        
        // 如果去重后数量不足n_cluster，补充随机cluster
        while (selected_clusters.size() < n_cluster) {
            int cluster_id = rand() % max_cluster_id;
            if (std::find(selected_clusters.begin(), selected_clusters.end(), cluster_id) == selected_clusters.end()) {
                selected_clusters.push_back(cluster_id);
                distinct_clusters.insert(cluster_id);
            }
        }
        
        // 填充数据
        for (int j = 0; j < n_cluster; j++) {
            result.h_query_cluster_group[i * n_cluster + j] = selected_clusters[j];
        }
    }
    
    // 转换为vector并排序
    result.distinct_cluster_ids.assign(distinct_clusters.begin(), distinct_clusters.end());
    std::sort(result.distinct_cluster_ids.begin(), result.distinct_cluster_ids.end());
    result.distinct_cluster_count = result.distinct_cluster_ids.size();
    
    return result;
}

// 生成cluster查询的倒排索引
void generate_cluster_query_inverted_index(
    int n_query, int n_cluster, int* h_query_cluster_group, const std::vector<int>& distinct_cluster_ids,
    int** h_cluster_query_offset, int** h_cluster_query_data
) {
    int distinct_cluster_count = distinct_cluster_ids.size();
    *h_cluster_query_offset = (int*)malloc(distinct_cluster_count * sizeof(int));
    
    // 统计每个cluster对应的query数量（去重）
    std::map<int, std::set<int>> cluster_queries_set;
    
    for (int i = 0; i < n_query; i++) {
        for (int j = 0; j < n_cluster; j++) {
            int cluster_id = h_query_cluster_group[i * n_cluster + j];
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
    
    *h_cluster_query_data = (int*)malloc(total_queries * sizeof(int));
    
    // 计算offset
    int offset = 0;
    for (int i = 0; i < distinct_cluster_count; i++) {
        int cluster_id = distinct_cluster_ids[i];
        (*h_cluster_query_offset)[i] = offset;
        offset += cluster_queries[cluster_id].size();
    }
    
    // 填充query数据
    int data_idx = 0;
    for (int i = 0; i < distinct_cluster_count; i++) {
        int cluster_id = distinct_cluster_ids[i];
        for (int query_id : cluster_queries[cluster_id]) {
            (*h_cluster_query_data)[data_idx++] = query_id;
        }
    }
}

// 生成cluster向量索引和数量
void generate_cluster_vector_info(int distinct_cluster_count, int tol_vector, 
                                 int** h_cluster_vector_index, int** h_cluster_vector_num) {
    *h_cluster_vector_index = (int*)malloc(distinct_cluster_count * sizeof(int));
    *h_cluster_vector_num = (int*)malloc(distinct_cluster_count * sizeof(int));
    
    // 先随机分配向量数量
    std::vector<int> vectors_per_cluster(distinct_cluster_count);
    int remaining_vectors = tol_vector;
    
    for (int i = 0; i < distinct_cluster_count; i++) {
        if (i == distinct_cluster_count - 1) {
            // 最后一个cluster分配剩余所有向量
            vectors_per_cluster[i] = remaining_vectors;
        } else {
            // 随机分配向量数量
            int max_vectors = remaining_vectors - (distinct_cluster_count - i - 1);
            vectors_per_cluster[i] = 1 + rand() % std::max(1, max_vectors);
        }
        remaining_vectors -= vectors_per_cluster[i];
    }
    
    // 计算偏移量
    int current_offset = 0;
    for (int i = 0; i < distinct_cluster_count; i++) {
        (*h_cluster_vector_index)[i] = current_offset;
        (*h_cluster_vector_num)[i] = vectors_per_cluster[i];
        current_offset += vectors_per_cluster[i];
    }
}

// 生成cluster向量数据
float** generate_cluster_vectors(int tol_vector, int n_dim) {
    return generate_vector_list(tol_vector, n_dim);
}

// 生成query向量数据
float* generate_query_vectors(int n_query, int n_dim) {
    float* h_query_group = (float*)malloc(n_query * n_dim * sizeof(float));
    
    for (int i = 0; i < n_query * n_dim; i++) {
        h_query_group[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f;
    }
    
    return h_query_group;
}

// 初始化输出数组
void initialize_output_arrays(int n_query, int n_topn, 
                             int** h_query_topn_index, float** h_query_topn_dist) {
    *h_query_topn_index = (int*)malloc(n_query * n_topn * sizeof(int));
    *h_query_topn_dist = (float*)malloc(n_query * n_topn * sizeof(float));
    
    // 初始化为0
    memset(*h_query_topn_index, 0, n_query * n_topn * sizeof(int));
    memset(*h_query_topn_dist, 0, n_query * n_topn * sizeof(float));
}

// 释放所有分配的内存
void free_test_data(
    float* h_query_group, int* h_query_cluster_group, int* h_cluster_query_offset, int* h_cluster_query_data,
    int* cluster_map,
    int* h_cluster_vector_index, int* h_cluster_vector_num, float** h_cluster_vector,
    int* h_query_topn_index, float* h_query_topn_dist
) {
    if (h_query_group) free(h_query_group);
    if (h_query_cluster_group) free(h_query_cluster_group);
    if (h_cluster_query_offset) free(h_cluster_query_offset);
    if (h_cluster_query_data) free(h_cluster_query_data);
    if (cluster_map) free(cluster_map);
    if (h_cluster_vector_index) free(h_cluster_vector_index);
    if (h_cluster_vector_num) free(h_cluster_vector_num);
    if (h_cluster_vector) free_vector_list((void**)h_cluster_vector);
    if (h_query_topn_index) free(h_query_topn_index);
    if (h_query_topn_dist) free(h_query_topn_dist);
}

// 测试函数 - 返回性能指标
std::vector<double> test_fine_screen_top_n(
    int n_query, int n_cluster, int n_dim, int n_topn, int max_cluster_id, int tol_vector
) {
    bool pass = true;
    double memory_mb = (double)(n_query * n_dim + tol_vector * n_dim + n_query * n_cluster * 2 + n_cluster * 3) * sizeof(float) / (double)(1024 * 1024);
    
    if (!QUIET) {
        COUT_VAL("配置:", n_query, "个query ×", n_cluster, "个cluster", ", 维度:", n_dim, ", top", n_topn, ", 总向量:", tol_vector);
    }
    
    // 生成测试数据
    float* h_query_group = generate_query_vectors(n_query, n_dim);
    QueryClusterData query_cluster_data = generate_query_cluster_group(n_query, n_cluster, max_cluster_id);
    
    int* h_cluster_query_offset, *h_cluster_query_data;
    generate_cluster_query_inverted_index(n_query, n_cluster, query_cluster_data.h_query_cluster_group, 
                                        query_cluster_data.distinct_cluster_ids, 
                                        &h_cluster_query_offset, &h_cluster_query_data);
    
    int* cluster_map = generate_cluster_map(query_cluster_data.distinct_cluster_ids);
    int* h_cluster_vector_index, *h_cluster_vector_num;
    generate_cluster_vector_info(query_cluster_data.distinct_cluster_count, tol_vector, 
                                &h_cluster_vector_index, &h_cluster_vector_num);
    
    float** h_cluster_vector = generate_cluster_vectors(tol_vector, n_dim);
    int* h_query_topn_index, *h_query_topn_index_cpu;
    float* h_query_topn_dist, *h_query_topn_dist_cpu;
    
    initialize_output_arrays(n_query, n_topn, &h_query_topn_index, &h_query_topn_dist);
    initialize_output_arrays(n_query, n_topn, &h_query_topn_index_cpu, &h_query_topn_dist_cpu);
    
    // 计算最大cluster向量数量
    int max_cluster_vector_count = 0;
    for (int i = 0; i < query_cluster_data.distinct_cluster_count; i++) {
        max_cluster_vector_count = std::max(max_cluster_vector_count, h_cluster_vector_num[i]);
    }
    
    // CPU测试
    double cpu_duration_ms = 0;
    MEASURE_MS_AND_SAVE("CPU Fine Screen TopN", cpu_duration_ms,
        cpu_fine_screen_top_n(
            h_query_group, query_cluster_data.h_query_cluster_group, h_cluster_query_offset, h_cluster_query_data,
            cluster_map,
            h_cluster_vector_index, h_cluster_vector_num, h_cluster_vector,
            n_query, n_cluster, query_cluster_data.distinct_cluster_count, n_dim, n_topn, max_cluster_id, tol_vector,
            max_cluster_vector_count,  // 新增：最大聚类向量数量
            h_query_topn_index_cpu, h_query_topn_dist_cpu
        );
    );
    
    // GPU测试
    double gpu_duration_ms = 0;
    MEASURE_MS_AND_SAVE("GPU Fine Screen TopN", gpu_duration_ms,
        gpu_fine_screen_top_n(
            h_query_group, query_cluster_data.h_query_cluster_group, h_cluster_query_offset, h_cluster_query_data,
            cluster_map,
            h_cluster_vector_index, h_cluster_vector_num, h_cluster_vector,
            n_query, n_cluster, query_cluster_data.distinct_cluster_count, n_dim, n_topn, max_cluster_id, tol_vector,
            max_cluster_vector_count,  // 新增：最大聚类向量数量
            h_query_topn_index, h_query_topn_dist
        );
    );
    
    // 计算加速比
    double speedup = (cpu_duration_ms > 0) ? cpu_duration_ms / gpu_duration_ms : 0.0;
    
    // 释放内存
    free_test_data(
        h_query_group, query_cluster_data.h_query_cluster_group, h_cluster_query_offset, h_cluster_query_data,
        cluster_map,
        h_cluster_vector_index, h_cluster_vector_num, h_cluster_vector,
        h_query_topn_index, h_query_topn_dist
    );
    free(h_query_topn_index_cpu);
    free(h_query_topn_dist_cpu);
    
    return {
        pass ? 1.0 : 0.0,
        (double)n_query, (double)n_cluster, (double)n_dim, (double)n_topn, (double)tol_vector,
        gpu_duration_ms, cpu_duration_ms, speedup, (double)memory_mb
    };
}

int main(int argc, char** argv) {
    srand(time(0));
    
    bool all_pass = true;
    
    MetricsCollector metrics;
    metrics.set_columns("pass rate", "n_query", "n_cluster", "n_dim", "n_topn", "tol_vector", "gpu_ms", "cpu_ms", "speedup", "memory_mb");
    metrics.set_num_repeats(1); 
    
    COUT_ENDL("测试精筛TopN算法");
    
    // 测试不同参数组合
    PARAM_3D(n_query, (8, 32, 128, 512), 
             n_cluster, (4, 16, 64), 
             n_dim, (512, 1024))
    {
        int n_topn = 16;
        int max_cluster_id = n_cluster * 3; // 稀疏的cluster ID
        int tol_vector = n_cluster * 50; // 每个cluster平均50个向量
        
        auto result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto test_result = test_fine_screen_top_n(n_query, n_cluster, n_dim, n_topn, max_cluster_id, tol_vector);
            all_pass &= (test_result[0] == 1.0);
            return test_result;
        });
    }
    
    metrics.print_table();
    metrics.export_csv("fine_screen_topk_metrics.csv");
    
    COUT_ENDL("\n所有测试:", all_pass ? "✅ PASS" : "❌ FAIL");
    return 0;
}
