#include <stdlib.h>
#include <limits>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

#include "../../cuda/fusion_cos_topk/fusion_cos_topk.cuh"
#include "../../cuda/pch.h"
#include "../common/test_utils.cuh"

#define EPSILON 1e-2f

enum AlgorithmVersion {
    WARP_SORT_FINE_V2,  // 融合余弦距离精筛（float4，一个block一个query）
    WARP_SORT_FINE_V3,  // 融合余弦距离精筛（一个block处理4个query）
    WARP_SORT_FINE_V3_32,  // 融合余弦距离精筛（一个block处理16个query）
    WARP_SORT_FINE_V4,  // 融合余弦距离精筛（混合策略：动态选择最优算法）
    ALL_VERSIONS        // 运行所有算法版本
};

typedef void (*CosineTopkFine)(
    float* d_query_vectors,
    float* d_cluster_vectors,
    int* d_query_index,
    
    float* d_query_norm,
    float* d_cluster_vector_norm,

    int* d_topk_index,
    float* d_topk_dist,

    int n_selected_querys,
    int n_selected_vectors,
    int n_dim,
    int k
);

struct AlgorithmInfo {
    std::string name;
    std::string description;
    CosineTopkFine func;
};

std::map<AlgorithmVersion, AlgorithmInfo> algorithm_registry = {
    {WARP_SORT_FINE_V2, {"warpsort_fine_v2", "WarpSort 精筛 v2 (一个block一个query)", cuda_cos_topk_warpsort_fine_v2}},
    {WARP_SORT_FINE_V3, {"warpsort_fine_v3", "WarpSort 精筛 v3 (一个block处理8个query)", cuda_cos_topk_warpsort_fine_v3}},
    {WARP_SORT_FINE_V3_32, {"warpsort_fine_v3_32", "WarpSort 精筛 v3_32 (一个block处理16个query)", cuda_cos_topk_warpsort_fine_v3_32}},
    {WARP_SORT_FINE_V4, {"warpsort_fine_v4", "WarpSort 精筛 v4 (混合策略：动态选择)", cuda_cos_topk_warpsort_fine_v4}}
};

/**
 * CPU版本余弦距离top-k计算（用于验证）
 */
void cpu_cos_distance_topk_fine(
    float** query_vectors,
    float** cluster_vectors,
    int* query_index,

    float* query_norm,
    float* cluster_vector_norm,
    
    int** topk_index,
    float** topk_dist,
    
    int n_selected_querys,
    int n_selected_vectors,
    int n_dim,
    int k
) {
    for (int q = 0; q < n_selected_querys; q++) {
        std::vector<std::pair<float, int>> candidates;

        float*  query = query_vectors[query_index[q]];

        for (int v = 0; v < n_selected_vectors; v++) {
            float dot_product = 0.0f;
            const float* vec_ptr = cluster_vectors[v];

            for (int d = 0; d < n_dim; d++) {
                dot_product += query[d] * vec_ptr[d];
            }

            float cos_distance = 1.0f - dot_product / (query_norm[query_index[q]] * cluster_vector_norm[v]);
            // if (denom > 1e-12f) {
            //     const float cos_similarity = dot_product / denom;
            //     cos_distance = 1.0f - cos_similarity;
            // }

            candidates.emplace_back(cos_distance, v);
        }

        std::sort(candidates.begin(), candidates.end());
        const int topk_count = std::min(k, static_cast<int>(candidates.size()));
        for (int i = 0; i < topk_count; i++) {
            topk_dist[q][i] = candidates[i].first;
            topk_index[q][i] = candidates[i].second;
        }
        for (int i = topk_count; i < k; i++) {
            topk_dist[q][i] = std::numeric_limits<float>::max();
            topk_index[q][i] = -1;
        }
    }
}

int* random_select(int n, int n_selected) {
    if (n_selected > n || n_selected <= 0) return NULL;
    
    int* result = (int*)malloc(n_selected * sizeof(int));
    if (!result) return NULL;
    
    int selected = 0;
    int remaining = n;
    
    for (int i = 0; i < n && selected < n_selected; i++) {
        if (rand() % remaining < n_selected - selected) {
            result[selected++] = i;
        }
        remaining--;
    }
    
    return result;
}

std::vector<double> test_fusion_cos_topk_fine_with_algorithm(
    int n_query,
    int n_selected_querys,
    int n_selected_vectors,
    int n_dim,
    int k,
    AlgorithmVersion algo_version
) {
    AlgorithmInfo& algo_info = algorithm_registry.at(algo_version);
    bool pass = true;

    if (!QUIET) {
        COUT_VAL("配置:", "n_query=", n_query,
                 " n_selected_querys=", n_selected_querys,
                 " n_vectors=", n_selected_vectors,
                 " n_dim=", n_dim,
                 " k=", k,
                 " 算法=", algo_info.description);
    }

    const size_t query_bytes = static_cast<size_t>(n_query) * n_dim * sizeof(float);
    const size_t vector_bytes = static_cast<size_t>(n_selected_vectors) * n_dim * sizeof(float);
    const size_t result_bytes = static_cast<size_t>(n_query) * k * (sizeof(int) + sizeof(float));

    float** h_query_vectors = generate_vector_list(n_query, n_dim);
    float** h_cluster_vectors = generate_vector_list(n_selected_vectors, n_dim);
    int* h_query_index = random_select(n_query, n_selected_querys);

    float* h_query_norm = (float*)malloc(n_query * sizeof(float));
    float* h_cluster_vector_norm = (float*)malloc(n_selected_vectors * sizeof(float));

    // 计算query向量的L2范数
    compute_l2_norms_batch(h_query_vectors, h_query_norm, n_query, n_dim);

    // 计算cluster向量的L2范数
    compute_l2_norms_batch(h_cluster_vectors, h_cluster_vector_norm, n_selected_vectors, n_dim);

    float *d_query_vectors = nullptr;
    float *d_cluster_vectors = nullptr;
    int *d_query_index = nullptr;
    float *d_query_norm = nullptr;
    float *d_cluster_vector_norm = nullptr;
    int *d_topk_index = nullptr;
    float *d_topk_dist = nullptr;

    cudaMalloc(&d_query_vectors, query_bytes);
    cudaMalloc(&d_cluster_vectors, vector_bytes);
    cudaMalloc(&d_query_index, n_selected_querys * sizeof(int));
    cudaMalloc(&d_query_norm, n_query * sizeof(float));
    cudaMalloc(&d_cluster_vector_norm, n_selected_vectors * sizeof(float));
    cudaMalloc(&d_topk_index, n_query * k * sizeof(int));
    cudaMalloc(&d_topk_dist, n_query * k * sizeof(float));
    CHECK_CUDA_ERRORS;

    cudaMemcpy(d_query_vectors, h_query_vectors[0], query_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vectors, h_cluster_vectors[0], vector_bytes, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;

    cudaMemcpy(d_query_index, h_query_index, n_selected_querys * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;

    cudaMemcpy(d_query_norm, h_query_norm, n_query * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vector_norm, h_cluster_vector_norm, n_selected_vectors * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;

    double gpu_duration_ms = 0.0;
    MEASURE_MS_AND_SAVE("gpu耗时:", gpu_duration_ms,
        algo_info.func(
            d_query_vectors,
            d_cluster_vectors,
            d_query_index,
            
            d_query_norm,
            d_cluster_vector_norm,
            
            d_topk_index,
            d_topk_dist,

            n_selected_querys,
            n_selected_vectors,
            n_dim,
            k
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );

    int** h_topk_index_gpu = (int**)malloc_vector_list(n_query, k, sizeof(int));
    float** h_topk_dist_gpu = (float**)malloc_vector_list(n_query, k, sizeof(float));
    cudaMemcpy(h_topk_index_gpu[0], d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_topk_dist_gpu[0], d_topk_dist, n_query * k * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;

    double cpu_duration_ms = 0.0;
    int** h_topk_index_cpu = (int**)malloc_vector_list(n_query, k, sizeof(int));
    float** h_topk_dist_cpu = (float**)malloc_vector_list(n_query, k, sizeof(float));

    MEASURE_MS_AND_SAVE("cpu耗时:", cpu_duration_ms,
        cpu_cos_distance_topk_fine(
            h_query_vectors,
            h_cluster_vectors,
            h_query_index,
            
            h_query_norm,
            h_cluster_vector_norm,
            
            h_topk_index_cpu,
            h_topk_dist_cpu,

            n_selected_querys,
            n_selected_vectors,
            n_dim,
            k
        );
    );

    pass &= compare_set_2D(h_topk_dist_gpu, h_topk_dist_cpu, n_selected_querys, k, EPSILON);

    cudaFree(d_query_vectors);
    cudaFree(d_cluster_vectors);
    cudaFree(d_query_index);
    cudaFree(d_query_norm);
    cudaFree(d_cluster_vector_norm);
    cudaFree(d_topk_index);
    cudaFree(d_topk_dist);
    CHECK_CUDA_ERRORS;

    const double speedup = (gpu_duration_ms > 1e-6) ? (cpu_duration_ms / gpu_duration_ms) : 0.0;
    const size_t aux_int_bytes = (static_cast<size_t>(n_query) + 1 + static_cast<size_t>(n_query) * k) * sizeof(int);
    const size_t aux_float_bytes = (static_cast<size_t>(n_query) + n_selected_vectors + static_cast<size_t>(n_query) * k) * sizeof(float);
    const double memory_mb = static_cast<double>(query_bytes + vector_bytes + result_bytes + aux_int_bytes + aux_float_bytes) / (1024.0 * 1024.0);

    return {
        pass ? 1.0 : 0.0,
        static_cast<double>(n_query),
        static_cast<double>(n_selected_querys),
        static_cast<double>(n_selected_vectors),
        static_cast<double>(n_dim),
        static_cast<double>(k),
        gpu_duration_ms,
        cpu_duration_ms,
        speedup,
        memory_mb
    };
}

AlgorithmVersion parse_algorithm_version(const char* arg) {
    const std::string version(arg);
    if (version == "v2" || version == "2") {
        return WARP_SORT_FINE_V2;
    }
    if (version == "v3" || version == "3") {
        return WARP_SORT_FINE_V3;
    }
    if (version == "v3_32" || version == "3_32") {
        return WARP_SORT_FINE_V3_32;
    }
    if (version == "v4" || version == "4") {
        return WARP_SORT_FINE_V4;
    }
    if (version == "warpsort" || version == "0") {
        return WARP_SORT_FINE_V2;  // 默认v2
    }
    if (version == "all" || version == "ALL" || version == "compare") {
        return ALL_VERSIONS;
    }
    COUT_VAL("未知的算法版本:", version.c_str());
    COUT_VAL("可用选项: v2, v3, v3_32, v4, all");
    exit(1);
}

bool run_algorithm_tests(AlgorithmVersion selected_version) {
    std::vector<AlgorithmVersion> versions;
    if (selected_version == ALL_VERSIONS) {
        for (const auto& kv : algorithm_registry) {
            versions.push_back(kv.first);
        }
    } else {
        versions.push_back(selected_version);
    }

    bool overall_pass = true;

    for (AlgorithmVersion version : versions) {
        AlgorithmInfo& info = algorithm_registry.at(version);
        COUT_ENDL("测试算法: ", info.description);

        MetricsCollector metrics;
        metrics.set_columns("pass rate", "n_query", "n_select_querys", "n_vectors", "n_dim", "k",
                             "avg_gpu_ms", "avg_cpu_ms", "avg_speedup", "memory_mb");

        test_fusion_cos_topk_fine_with_algorithm(
            10000,
            10,
            1024,
            128,
            100,
            version
        );

        // metrics.set_num_repeats(50);
        // PARAM_3D(n_selected_querys, (1, 5, 10, 100, 128, 200, 256, 300, 384, 512, 600, 1000, 1200),
        //          n_selected_vectors, (1024),
        //          n_dim, (128))
        
        // metrics.set_num_repeats(1);        
        // PARAM_3D(n_selected_vectors, (128, 1000, 10000),
        //          n_selected_querys, (100, 1000),
        //          n_dim, (128, 512, 1024))

        metrics.set_num_repeats(1);
        PARAM_3D(n_selected_vectors, (1, 2, 10, 20, 50, 90, 95, 99, 100, 101, 128, 1000, 10000),
            n_selected_querys, (1, 5, 10, 100, 128, 200, 256, 300, 384, 512, 600, 1000, 1200),
            n_dim, (128))
        {
            // int n_selected_vectors = 10000;
            // int k = 100;
            int n_query = 10000;
            // int n_selected_vectors = 10000;
            // int k = 100;
            int k = 2;

            auto avg_result = metrics.add_row_averaged([&]() -> std::vector<double> {
                auto result = test_fusion_cos_topk_fine_with_algorithm(
                    n_query,
                    n_selected_querys,
                    n_selected_vectors,
                    n_dim,
                    k,
                    version
                );
                overall_pass &= (result[0] == 1.0);
                return result;
            });

            if (!QUIET) {
                COUT_VAL("结果 (pass rate=", avg_result[0], ", speedup=", avg_result[8], ")");
            }
        }

        metrics.print_table();
        metrics.export_csv(info.name + "_metrics.csv");
    }

    return overall_pass;
}

int main(int argc, char** argv) {
    srand(static_cast<unsigned>(time(nullptr)));

    AlgorithmVersion selected_version = WARP_SORT_FINE_V3;
    // AlgorithmVersion selected_version = WARP_SORT_FINE_V2;
    if (argc > 1) {
        selected_version = parse_algorithm_version(argv[1]);
    }

    bool all_pass = run_algorithm_tests(selected_version);
    COUT_ENDL("\n所有测试:", (all_pass ? " ✅ PASS" : " ❌ FAIL"));        

    return all_pass ? 0 : 1;
}
