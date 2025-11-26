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

#include "../../cuda/fusion_cos_topk/fusion_cos_topk.cuh"
#include "../../cuda/pch.h"
#include "../common/test_utils.cuh"
#include "../common/params_macros.cuh"
#include "../common/output_macros.cuh"

#define EPSILON 1e-2f

/**
 * CPU版本余弦距离top-k计算（用于验证）
 * 
 * 对于 fixed_probe 版本，需要：
 * 1. 对每个 query，遍历所有 probe
 * 2. 对每个 probe，遍历该 probe 的所有向量
 * 3. 计算余弦距离并选择 top-k
 * 4. 合并所有 probe 的结果，选择全局 top-k
 */
void cpu_cos_distance_topk_fine_v3_fixed_probe(
    float** query_vectors,
    float** cluster_vectors,
    int* probe_vector_offset,
    int* probe_vector_count,
    int* probe_queries,
    int* probe_query_offsets,
    int* probe_query_probe_indices,
    float* query_norm,
    float* cluster_vector_norm,
    int** topk_index,
    float** topk_dist,
    int n_query,
    int n_probes,
    int n_dim,
    int k
) {
    // 为每个 query 收集所有候选
    for (int q = 0; q < n_query; q++) {
        std::vector<std::pair<float, int>> candidates;
        
        // 遍历所有 probe
        for (int p = 0; p < n_probes; p++) {
            // 检查该 probe 是否包含当前 query
            int probe_start = probe_query_offsets[p];
            int probe_end = probe_query_offsets[p + 1];
            bool query_in_probe = false;
            int probe_index_in_query = -1;
            
            for (int idx = probe_start; idx < probe_end; idx++) {
                if (probe_queries[idx] == q) {
                    query_in_probe = true;
                    probe_index_in_query = probe_query_probe_indices[idx];
                    break;
                }
            }
            
            if (!query_in_probe) continue;
            
            // 遍历该 probe 的所有向量
            int vec_start = probe_vector_offset[p];
            int vec_end = vec_start + probe_vector_count[p];
            
            float* query = query_vectors[q];
            float query_n = query_norm[q];
            
            for (int v = vec_start; v < vec_end; v++) {
                float dot_product = 0.0f;
                const float* vec_ptr = cluster_vectors[v];
                
                for (int d = 0; d < n_dim; d++) {
                    dot_product += query[d] * vec_ptr[d];
                }
                
                float vec_n = cluster_vector_norm[v];
                if (vec_n < 1e-6f || query_n < 1e-6f) continue;
                
                float cos_similarity = dot_product / sqrt(query_n * vec_n);
                float cos_distance = 1.0f - cos_similarity;
                
                candidates.emplace_back(cos_distance, v);
            }
        }
        
        // 排序并选择 top-k
        std::sort(candidates.begin(), candidates.end());
        const int topk_count = std::min(k, static_cast<int>(candidates.size()));
        for (int i = 0; i < topk_count; i++) {
            topk_dist[q][i] = candidates[i].first;
            topk_index[q][i] = candidates[i].second;
        }
        for (int i = topk_count; i < k; i++) {
            topk_dist[q][i] = FLT_MAX;
            topk_index[q][i] = -1;
        }
    }
}

/**
 * 测试单个参数组合
 */
std::vector<double> test_single_config(int n_query, int n_probes, int n_dim, int k, int vectors_per_probe) {
    // 1. 生成测试数据
    float** h_query_vectors = generate_vector_list(n_query, n_dim);
    int n_total_vectors = n_probes * vectors_per_probe;
    float** h_cluster_vectors = generate_vector_list(n_total_vectors, n_dim);
    
    // 2. 计算 L2 范数（平方和，用于后续计算）
    float* h_query_norm = (float*)malloc(n_query * sizeof(float));
    float* h_cluster_vector_norm = (float*)malloc(n_total_vectors * sizeof(float));
    
    // 计算query向量的平方和
    compute_squared_sums_batch(h_query_vectors, h_query_norm, n_query, n_dim);
    
    // 计算cluster向量的平方和
    compute_squared_sums_batch(h_cluster_vectors, h_cluster_vector_norm, n_total_vectors, n_dim);
    
    // 3. 构建 probe-query 映射（CSR 格式）
    // 每个 probe 包含所有 query（简化测试）
    std::vector<int> probe_query_offsets;
    std::vector<int> probe_queries;
    std::vector<int> probe_query_probe_indices;
    
    probe_query_offsets.push_back(0);
    for (int p = 0; p < n_probes; p++) {
        for (int q = 0; q < n_query; q++) {
            probe_queries.push_back(q);
            probe_query_probe_indices.push_back(p);  // probe 在 query 中的索引就是 p
        }
        probe_query_offsets.push_back(probe_queries.size());
    }
    
    // 4. 构建 probe 向量映射
    std::vector<int> probe_vector_offset;
    std::vector<int> probe_vector_count;
    for (int p = 0; p < n_probes; p++) {
        probe_vector_offset.push_back(p * vectors_per_probe);
        probe_vector_count.push_back(vectors_per_probe);
    }
    
    // 5. CPU 参考实现
    float** h_topk_dist_cpu = (float**)malloc_vector_list(n_query, k, sizeof(float));
    int** h_topk_index_cpu = (int**)malloc_vector_list(n_query, k, sizeof(int));
    
    double cpu_duration_ms = 0;
    MEASURE_MS_AND_SAVE("CPU fine search", cpu_duration_ms,
        cpu_cos_distance_topk_fine_v3_fixed_probe(
            h_query_vectors,
            h_cluster_vectors,
            probe_vector_offset.data(),
            probe_vector_count.data(),
            probe_queries.data(),
            probe_query_offsets.data(),
            probe_query_probe_indices.data(),
            h_query_norm,
            h_cluster_vector_norm,
            h_topk_index_cpu,
            h_topk_dist_cpu,
            n_query,
            n_probes,
            n_dim,
            k
        );
    );
    
    // 6. GPU 实现
    float* d_query_group = nullptr;
    float* d_cluster_vector = nullptr;
    int* d_probe_vector_offset = nullptr;
    int* d_probe_vector_count = nullptr;
    int* d_probe_queries = nullptr;
    int* d_probe_query_offsets = nullptr;
    int* d_probe_query_probe_indices = nullptr;
    float* d_query_norm = nullptr;
    float* d_cluster_vector_norm = nullptr;
    int* d_topk_index = nullptr;
    float* d_topk_dist = nullptr;
    
    // 分配设备内存
    cudaMalloc(&d_query_group, n_query * n_dim * sizeof(float));
    cudaMalloc(&d_cluster_vector, n_total_vectors * n_dim * sizeof(float));
    cudaMalloc(&d_probe_vector_offset, n_probes * sizeof(int));
    cudaMalloc(&d_probe_vector_count, n_probes * sizeof(int));
    cudaMalloc(&d_probe_queries, probe_queries.size() * sizeof(int));
    cudaMalloc(&d_probe_query_offsets, (n_probes + 1) * sizeof(int));
    cudaMalloc(&d_probe_query_probe_indices, probe_query_probe_indices.size() * sizeof(int));
    cudaMalloc(&d_query_norm, n_query * sizeof(float));
    cudaMalloc(&d_cluster_vector_norm, n_total_vectors * sizeof(float));
    cudaMalloc(&d_topk_index, n_query * k * sizeof(int));
    cudaMalloc(&d_topk_dist, n_query * k * sizeof(float));
    CHECK_CUDA_ERRORS;
    
    // 复制数据到设备
    cudaMemcpy(d_query_group, h_query_vectors[0], n_query * n_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vector, h_cluster_vectors[0], n_total_vectors * n_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_vector_offset, probe_vector_offset.data(), n_probes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_vector_count, probe_vector_count.data(), n_probes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_queries, probe_queries.data(), probe_queries.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_query_offsets, probe_query_offsets.data(), (n_probes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_query_probe_indices, probe_query_probe_indices.data(), probe_query_probe_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_norm, h_query_norm, n_query * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vector_norm, h_cluster_vector_norm, n_total_vectors * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;
    
    // 准备候选结果缓冲区（用于调试输出）
    float** candidate_dist = (float**)malloc_vector_list(n_query, n_probes * k, sizeof(float));
    int** candidate_index = (int**)malloc_vector_list(n_query, n_probes * k, sizeof(int));
    
    // GPU kernel 执行
    double gpu_duration_ms = 0;
    MEASURE_MS_AND_SAVE("GPU fine search", gpu_duration_ms,
        cuda_cos_topk_warpsort_fine_v3_fixed_probe(
            d_query_group,
            d_cluster_vector,
            d_probe_vector_offset,
            d_probe_vector_count,
            d_probe_queries,
            d_probe_query_offsets,
            d_probe_query_probe_indices,
            d_query_norm,
            d_cluster_vector_norm,
            d_topk_index,
            d_topk_dist,
            candidate_dist,
            candidate_index,
            n_query,
            n_probes,
            n_dim,
            k
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );
    
    // 复制结果回主机
    float** h_topk_dist_gpu = (float**)malloc_vector_list(n_query, k, sizeof(float));
    int** h_topk_index_gpu = (int**)malloc_vector_list(n_query, k, sizeof(int));
    
    cudaMemcpy(h_topk_dist_gpu[0], d_topk_dist, n_query * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_topk_index_gpu[0], d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;
    
    // 7. 验证结果（使用 test_utils.cuh 的 compare_set_2D）
    bool pass = true;
    pass &= compare_set_2D(h_topk_dist_gpu, h_topk_dist_cpu, n_query, k, EPSILON);
    // 注意：索引比较可能因为距离相同而顺序不同，这里只比较距离
    // 如果需要比较索引，可以使用 compare_set_2D(h_topk_index_gpu, h_topk_index_cpu, n_query, k)
    
    // 8. 清理
    cudaFree(d_query_group);
    cudaFree(d_cluster_vector);
    cudaFree(d_probe_vector_offset);
    cudaFree(d_probe_vector_count);
    cudaFree(d_probe_queries);
    cudaFree(d_probe_query_offsets);
    cudaFree(d_probe_query_probe_indices);
    cudaFree(d_query_norm);
    cudaFree(d_cluster_vector_norm);
    cudaFree(d_topk_index);
    cudaFree(d_topk_dist);
    
    free_vector_list((void**)h_query_vectors);
    free_vector_list((void**)h_cluster_vectors);
    free_vector_list((void**)h_topk_dist_cpu);
    free_vector_list((void**)h_topk_index_cpu);
    free_vector_list((void**)h_topk_dist_gpu);
    free_vector_list((void**)h_topk_index_gpu);
    free_vector_list((void**)candidate_dist);
    free_vector_list((void**)candidate_index);
    free(h_query_norm);
    free(h_cluster_vector_norm);
    
    // 9. 返回结果
    double pass_rate = pass ? 1.0 : 0.0;
    double speedup = cpu_duration_ms > 0 ? cpu_duration_ms / gpu_duration_ms : 0.0;
    double memory_mb = (double)(n_query * n_dim + n_total_vectors * n_dim) * sizeof(float) / (double)(1024 * 1024);
    
    return {pass_rate, (double)n_query, (double)n_probes, (double)n_dim, (double)k, 
            gpu_duration_ms, cpu_duration_ms, speedup, memory_mb};
}

int main(int argc, char** argv) {
    srand(time(0));
    
    bool all_pass = true;
    MetricsCollector metrics;
    metrics.set_columns("pass rate", "n_query", "n_probes", "n_dim", "k", 
                        "avg_gpu_ms", "avg_cpu_ms", "avg_speedup", "memory_mb");
    
    // Warmup
    test_single_config(8, 4, 128, 10, 32);
    
    COUT_ENDL("测试算法: cuda_cos_topk_warpsort_fine_v3_fixed_probe");
    
    // 参数扫描
    // PARAM_3D(n_query, (8, 32, 128),
    //          n_probes, (4, 8, 16),
    //          k, (10, 20))
    PARAM_3D(n_query, (10000),
        n_probes, (1024),
        k, (100))
    {
        int n_dim = 128;
        // int vectors_per_probe = 32;  // 固定每个 probe 的向量数
        int vectors_per_probe = 1024;  // 固定每个 probe 的向量数
        auto avg_result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto result = test_single_config(n_query, n_probes, n_dim, k, vectors_per_probe);
            all_pass &= (result[0] == 1.0);  // 检查 pass 字段
            return result;
        });
    }
    
    metrics.print_table();
    
    COUT_ENDL(all_pass ? "✅ All tests passed!" : "❌ Some tests failed!");
    return all_pass ? 0 : 1;
}

