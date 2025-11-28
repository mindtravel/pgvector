#include <stdlib.h>
#include <limits>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <ctime>
#include <cfloat>
#include <cmath>
#include <random>
#include <thread>

#include "../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort_count.cuh"
#include "../../cuda/pch.h"
#include "../common/test_utils.cuh"
#include "../common/params_macros.cuh"
#include "../common/output_macros.cuh"

#define EPSILON 1e-2f

/**
 * CPU版本粗筛计算（用于验证）
 * 计算query和cluster中心之间的余弦距离，选择topk
 */
void cpu_cos_topk_warpsort_count(
    float** h_query_vectors,
    float** h_data_vectors,  // cluster中心向量
    float* h_data_norm,
    int** h_topk_index,
    int* h_cluster_query_count,
    int* h_cluster_query_offset,
    int* h_cluster_query_data,
    int* h_cluster_query_probe_indices,
    int n_query,
    int n_total_clusters,
    int n_dim,
    int k
) {
    // 1. 计算query的L2范数
    float* h_query_norm = (float*)malloc(n_query * sizeof(float));
    compute_l2_norms_batch(h_query_vectors, h_query_norm, n_query, n_dim);
    
    // 2. 计算每个query的topk cluster
    for (int q = 0; q < n_query; q++) {
        std::vector<std::pair<float, int>> candidates;
        
        for (int c = 0; c < n_total_clusters; c++) {
            float dot_product = 0.0f;
            for (int d = 0; d < n_dim; d++) {
                dot_product += h_query_vectors[q][d] * h_data_vectors[c][d];
            }
            
            float query_n = h_query_norm[q];
            float data_n = h_data_norm[c];
            
            if (query_n < 1e-6f || data_n < 1e-6f) continue;
            
            float cos_similarity = dot_product / (query_n * data_n);
            float cos_distance = 1.0f - cos_similarity;
            
            candidates.emplace_back(cos_distance, c);
        }
        
        // 使用partial_sort选择topk
        int topk_count = std::min(k, static_cast<int>(candidates.size()));
        if (topk_count > 0) {
            std::partial_sort(
                candidates.begin(),
                candidates.begin() + topk_count,
                candidates.end()
            );
            
            for (int i = 0; i < topk_count; i++) {
                h_topk_index[q][i] = candidates[i].second;
            }
        }
        // 填充剩余位置为无效值
        for (int i = topk_count; i < k; i++) {
            h_topk_index[q][i] = -1;
        }
    }
    
    // 3. 构建cluster-query映射（CSR格式）
    // 第一步：统计每个cluster有多少个query使用它
    for (int c = 0; c < n_total_clusters; c++) {
        h_cluster_query_count[c] = 0;
    }
    
    for (int q = 0; q < n_query; q++) {
        for (int rank = 0; rank < k; rank++) {
            int cluster_id = h_topk_index[q][rank];
            if (cluster_id >= 0 && cluster_id < n_total_clusters) {
                h_cluster_query_count[cluster_id]++;
            }
        }
    }
    
    // 第二步：构建CSR格式的offset数组
    h_cluster_query_offset[0] = 0;
    for (int c = 0; c < n_total_clusters; c++) {
        h_cluster_query_offset[c + 1] = h_cluster_query_offset[c] + h_cluster_query_count[c];
    }
    
    // 第三步：填充CSR格式的数据
    int* cluster_write_pos = (int*)malloc(n_total_clusters * sizeof(int));
    for (int c = 0; c < n_total_clusters; c++) {
        cluster_write_pos[c] = h_cluster_query_offset[c];
    }
    
    for (int q = 0; q < n_query; q++) {
        for (int rank = 0; rank < k; rank++) {
            int cluster_id = h_topk_index[q][rank];
            if (cluster_id < 0 || cluster_id >= n_total_clusters) continue;
            
            int write_pos = cluster_write_pos[cluster_id];
            h_cluster_query_data[write_pos] = q;
            h_cluster_query_probe_indices[write_pos] = rank;
            cluster_write_pos[cluster_id]++;
        }
    }
    
    free(cluster_write_pos);
    free(h_query_norm);
}

/**
 * 测试单个参数组合
 */
std::vector<double> test_single_config(
    int n_query, 
    int n_total_clusters, 
    int n_dim, 
    int k
) {
    // 1. 生成测试数据
    float** h_query_vectors = generate_vector_list(n_query, n_dim);
    float** h_data_vectors = generate_vector_list(n_total_clusters, n_dim);  // cluster中心
    
    // 计算data向量的L2范数
    float* h_data_norm = (float*)malloc(n_total_clusters * sizeof(float));
    compute_l2_norms_batch(h_data_vectors, h_data_norm, n_total_clusters, n_dim);
    
    // 2. CPU参考实现
    int** h_topk_index_cpu = (int**)malloc_vector_list(n_query, k, sizeof(int));
    int* h_cluster_query_count_cpu = (int*)calloc(n_total_clusters, sizeof(int));
    
    int total_entries_estimate = n_query * k;  // 最大可能的entries数
    int* h_cluster_query_offset_cpu = (int*)malloc((n_total_clusters + 1) * sizeof(int));
    int* h_cluster_query_data_cpu = (int*)malloc(total_entries_estimate * sizeof(int));
    int* h_cluster_query_probe_indices_cpu = (int*)malloc(total_entries_estimate * sizeof(int));
    
    double cpu_duration_ms = 0;
    MEASURE_MS_AND_SAVE("CPU coarse search", cpu_duration_ms,
        cpu_cos_topk_warpsort_count(
            h_query_vectors,
            h_data_vectors,
            h_data_norm,
            h_topk_index_cpu,
            h_cluster_query_count_cpu,
            h_cluster_query_offset_cpu,
            h_cluster_query_data_cpu,
            h_cluster_query_probe_indices_cpu,
            n_query,
            n_total_clusters,
            n_dim,
            k
        );
    );
    
    int total_entries_cpu = h_cluster_query_offset_cpu[n_total_clusters];
    
    // 3. GPU实现
    int** h_topk_index_gpu = (int**)malloc_vector_list(n_query, k, sizeof(int));
    
    // 分配GPU内存
    float* d_data_norm = nullptr;
    int* d_cluster_query_count = nullptr;
    int* d_cluster_query_offset = nullptr;
    int* d_cluster_query_data = nullptr;
    int* d_cluster_query_probe_indices = nullptr;
    ClusterQueryGroup* d_compact_groups = nullptr;
    int* d_n_groups = nullptr;
    int h_n_groups = 0;
    
    cudaMalloc(&d_data_norm, n_total_clusters * sizeof(float));
    cudaMalloc(&d_cluster_query_count, n_total_clusters * sizeof(int));
    cudaMalloc(&d_cluster_query_offset, (n_total_clusters + 1) * sizeof(int));
    cudaMalloc(&d_cluster_query_data, total_entries_estimate * sizeof(int));
    cudaMalloc(&d_cluster_query_probe_indices, total_entries_estimate * sizeof(int));
    
    // 估计最大组数量（每组4个query）
    int max_groups = (total_entries_estimate + 3) / 4;
    cudaMalloc(&d_compact_groups, max_groups * sizeof(ClusterQueryGroup));
    cudaMalloc(&d_n_groups, sizeof(int));
    
    // 复制data_norm到GPU
    cudaMemcpy(d_data_norm, h_data_norm, n_total_clusters * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;
    
    // GPU kernel执行
    double gpu_duration_ms = 0;
    MEASURE_MS_AND_SAVE("GPU coarse search", gpu_duration_ms,
        cuda_cos_topk_warpsort_count(
            h_query_vectors,
            h_data_vectors,
            d_data_norm,
            h_topk_index_gpu,
            d_cluster_query_count,
            d_cluster_query_offset,
            d_cluster_query_data,
            d_cluster_query_probe_indices,
            d_compact_groups,
            d_n_groups,
            n_query,
            n_total_clusters,
            n_dim,
            k
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );
    
    // 复制结果回主机
    int* h_cluster_query_count_gpu = (int*)malloc(n_total_clusters * sizeof(int));
    int* h_cluster_query_offset_gpu = (int*)malloc((n_total_clusters + 1) * sizeof(int));
    int* h_cluster_query_data_gpu = (int*)malloc(total_entries_estimate * sizeof(int));
    int* h_cluster_query_probe_indices_gpu = (int*)malloc(total_entries_estimate * sizeof(int));
    ClusterQueryGroup* h_compact_groups = (ClusterQueryGroup*)malloc(max_groups * sizeof(ClusterQueryGroup));
    
    cudaMemcpy(h_cluster_query_count_gpu, d_cluster_query_count, n_total_clusters * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cluster_query_offset_gpu, d_cluster_query_offset, (n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_n_groups, d_n_groups, sizeof(int), cudaMemcpyDeviceToHost);  // 读取n_groups
    
    int total_entries_gpu = h_cluster_query_offset_gpu[n_total_clusters];
    if (total_entries_gpu > 0) {
        cudaMemcpy(h_cluster_query_data_gpu, d_cluster_query_data, total_entries_gpu * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cluster_query_probe_indices_gpu, d_cluster_query_probe_indices, total_entries_gpu * sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    if (h_n_groups > 0) {
        cudaMemcpy(h_compact_groups, d_compact_groups, h_n_groups * sizeof(ClusterQueryGroup), cudaMemcpyDeviceToHost);
    }
    
    CHECK_CUDA_ERRORS;
    
    // 4. 验证结果
    bool pass = true;
    
    // 验证topk索引（使用集合比较，忽略顺序）
    pass &= compare_set_2D(h_topk_index_gpu, h_topk_index_cpu, n_query, k, 0.0f);
    
    // 验证cluster_query_count
    bool count_match = true;
    for (int c = 0; c < n_total_clusters; c++) {
        if (h_cluster_query_count_gpu[c] != h_cluster_query_count_cpu[c]) {
            if (count_match) {
                printf("[ERROR] cluster_query_count mismatch:\n");
                count_match = false;
            }
            printf("  Cluster %d: GPU=%d, CPU=%d\n", c, h_cluster_query_count_gpu[c], h_cluster_query_count_cpu[c]);
        }
    }
    pass &= count_match;
    
    // 验证cluster_query_offset
    bool offset_match = true;
    for (int c = 0; c <= n_total_clusters; c++) {
        if (h_cluster_query_offset_gpu[c] != h_cluster_query_offset_cpu[c]) {
            if (offset_match) {
                printf("[ERROR] cluster_query_offset mismatch:\n");
                offset_match = false;
            }
            printf("  Offset[%d]: GPU=%d, CPU=%d\n", c, h_cluster_query_offset_gpu[c], h_cluster_query_offset_cpu[c]);
        }
    }
    pass &= offset_match;
    
    // 验证cluster_query_data和probe_indices（需要按cluster分组比较）
    bool data_match = true;
    if (total_entries_gpu == total_entries_cpu) {
        // 为每个cluster构建集合进行比较
        for (int c = 0; c < n_total_clusters; c++) {
            int start = h_cluster_query_offset_cpu[c];
            int end = h_cluster_query_offset_cpu[c + 1];
            
            std::set<std::pair<int, int>> cpu_pairs;
            std::set<std::pair<int, int>> gpu_pairs;
            
            for (int i = start; i < end; i++) {
                cpu_pairs.insert({h_cluster_query_data_cpu[i], h_cluster_query_probe_indices_cpu[i]});
                gpu_pairs.insert({h_cluster_query_data_gpu[i], h_cluster_query_probe_indices_gpu[i]});
            }
            
            if (cpu_pairs != gpu_pairs) {
                if (data_match) {
                    printf("[ERROR] cluster_query_data/probe_indices mismatch:\n");
                    data_match = false;
                }
                printf("  Cluster %d: CPU pairs=%zu, GPU pairs=%zu\n", c, cpu_pairs.size(), gpu_pairs.size());
            }
        }
    } else {
        printf("[ERROR] total_entries mismatch: GPU=%d, CPU=%d\n", total_entries_gpu, total_entries_cpu);
        data_match = false;
    }
    pass &= data_match;
    
    // 验证紧凑格式的组数据
    bool compact_match = true;
    if (h_n_groups > 0) {
        // 从紧凑格式重建cluster-query映射，与CSR格式比较
        std::map<int, std::set<std::pair<int, int>>> compact_pairs;
        
        for (int g = 0; g < h_n_groups; g++) {
            int cluster_id = h_compact_groups[g].cluster_id;
            for (int i = 0; i < 4; i++) {
                int query_id = h_compact_groups[g].query_ids[i];
                int probe_idx = h_compact_groups[g].probe_indices[i];
                
                if (query_id >= 0 && probe_idx >= 0) {
                    compact_pairs[cluster_id].insert({query_id, probe_idx});
                }
            }
        }
        
        // 与CSR格式比较
        for (int c = 0; c < n_total_clusters; c++) {
            int start = h_cluster_query_offset_cpu[c];
            int end = h_cluster_query_offset_cpu[c + 1];
            
            std::set<std::pair<int, int>> csr_pairs;
            for (int i = start; i < end; i++) {
                csr_pairs.insert({h_cluster_query_data_cpu[i], h_cluster_query_probe_indices_cpu[i]});
            }
            
            if (compact_pairs[c] != csr_pairs) {
                if (compact_match) {
                    printf("[ERROR] compact groups mismatch:\n");
                    compact_match = false;
                }
                printf("  Cluster %d: CSR pairs=%zu, Compact pairs=%zu\n", c, csr_pairs.size(), compact_pairs[c].size());
            }
        }
    }
    pass &= compact_match;
    
    // 5. 清理
    cudaFree(d_data_norm);
    cudaFree(d_cluster_query_count);
    cudaFree(d_cluster_query_offset);
    cudaFree(d_cluster_query_data);
    cudaFree(d_cluster_query_probe_indices);
    cudaFree(d_compact_groups);
    cudaFree(d_n_groups);
    
    free_vector_list((void**)h_query_vectors);
    free_vector_list((void**)h_data_vectors);
    free(h_data_norm);
    free_vector_list((void**)h_topk_index_cpu);
    free(h_cluster_query_count_cpu);
    free(h_cluster_query_offset_cpu);
    free(h_cluster_query_data_cpu);
    free(h_cluster_query_probe_indices_cpu);
    free_vector_list((void**)h_topk_index_gpu);
    free(h_cluster_query_count_gpu);
    free(h_cluster_query_offset_gpu);
    free(h_cluster_query_data_gpu);
    free(h_cluster_query_probe_indices_gpu);
    free(h_compact_groups);
    
    // 6. 返回结果
    double pass_rate = pass ? 1.0 : 0.0;
    double speedup = cpu_duration_ms > 0 ? cpu_duration_ms / gpu_duration_ms : 0.0;
    
    return {pass_rate, static_cast<double>(n_query), static_cast<double>(n_total_clusters), 
            static_cast<double>(n_dim), static_cast<double>(k), 
            gpu_duration_ms, cpu_duration_ms, speedup};
}

int main(int argc, char** argv) {
    MetricsCollector metrics;
    metrics.set_columns("pass_rate", "n_query", "n_clusters", "vector_dim", 
                       "k", "gpu_ms", "cpu_ms", "speedup");
    metrics.set_num_repeats(1);
    
    // 测试参数
    PARAM_3D(n_query, (4, 10, 100),
        n_total_clusters, (10, 50, 200),
        vector_dim, (32, 128, 256))
    {
        int k = 5;
        
        if (!QUIET) {
            COUT_ENDL("========================================");
            COUT_VAL("Testing: n_query=", n_query,
                    " n_clusters=", n_total_clusters,
                    " vector_dim=", vector_dim,
                    " k=", k);
            COUT_ENDL("========================================");
        }
        
        auto result = test_single_config(n_query, n_total_clusters, vector_dim, k);
        // add_row 需要展开 vector 为多个参数
        metrics.add_row(
            result[0],  // pass_rate
            result[1],  // n_query
            result[2],  // n_clusters
            result[3],  // vector_dim
            result[4],  // k
            result[5],  // gpu_ms
            result[6],  // cpu_ms
            result[7]   // speedup
        );
        
        if (!QUIET) {
            COUT_VAL("Result: pass_rate=", result[0],
                    " gpu_ms=", result[5],
                    " cpu_ms=", result[6],
                    " speedup=", result[7]);
            COUT_ENDL();
        }
    };
    // 打印统计表格
    metrics.print_table();
    
    // 导出 CSV
    metrics.export_csv("fusion_cos_topk_warpsort_count_metrics.csv");
    
    if (!QUIET) {
        COUT_ENDL("All tests completed!");
    }
    
    return 0;
}

