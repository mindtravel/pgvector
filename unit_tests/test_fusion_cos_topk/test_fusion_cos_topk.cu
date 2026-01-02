#include <stdlib.h>
#include <limits>
#include <string>
#include <map>
#include <thread>
#include <algorithm>

#include "../cuda/fusion_dist_topk/fusion_dist_topk.cuh"
#include "../cuda/pch.h"
#include "../common/test_utils.cuh"
#include "../cpu_utils/cpu_utils.h"

#define EPSILON 1e-2
#define DIV_EPSILON 1e-4

// 算法版本枚举
enum AlgorithmVersion {
    WARP_SORT,          // 纯寄存器 topk
    ALL_VERSIONS        // 运行所有版本
};

// 算法函数指针类型定义
typedef void (*CosineTopkFunc)(
    const float** query_vectors, const float** data_vectors, 
    int** topk_index, float** topk_dist,
    int n_query, int n_batch, int n_dim, int k
);

// 算法版本信息结构
struct AlgorithmInfo {
    std::string name;
    std::string description;
    CosineTopkFunc func;
};

// 算法注册表
std::map<AlgorithmVersion, AlgorithmInfo> algorithm_registry;

// 初始化算法注册表
void init_algorithm_registry() {
    AlgorithmInfo info = {"warpsort", "纯寄存器 Top k", cuda_cos_topk_warpsort};
    algorithm_registry[WARP_SORT] = info;
}


/**
 * 测试函数 - 返回所有性能指标
 * @param n_query 查询向量数量
 * @param n_batch 数据向量数量
 * @param n_dim 向量维度
 * @param k top-k值
 * @param algo_version 算法版本
 * @param h_query_vectors 预生成的查询向量（如果为nullptr，则内部生成）
 * @param h_data_vectors 预生成的数据向量（如果为nullptr，则内部生成）
 * @return vector<double>: {pass rate, n_query, n_batch, n_dim, k, gpu_ms, cpu_ms, speedup, memery_mb}
 */
std::vector<double> test_cos_distance_topk_with_algorithm(
    int n_query, int n_batch, int n_dim, int k, 
    AlgorithmVersion algo_version,
    const float** h_query_vectors = nullptr,
    const float** h_data_vectors = nullptr
) 
{
    AlgorithmInfo& algo_info = algorithm_registry[algo_version];
    bool pass = true;
    double memory_mb = (double)(n_query * n_dim + n_batch * n_dim + n_query * n_batch) * sizeof(float) / (double)(1024 * 1024); /* 内存使用量 */
    
    if (!QUIET) {
        COUT_VAL("配置:", n_query, "个查询向量 ×", n_batch, "个数据向量", ", 向量维度:", n_dim, ", 找top", k, ", 算法: ", algo_info.description);
    }  
        
    // 生成测试数据（如果未提供）
    bool need_free_query = false;
    bool need_free_data = false;
    if (h_query_vectors == nullptr) {
        h_query_vectors = generate_vector_list<const float>(n_query, n_dim);
        need_free_query = true;
    }
    if (h_data_vectors == nullptr) {
        h_data_vectors = generate_vector_list<const float>(n_batch, n_dim);
        need_free_data = true;
    }
    int** topk_index_cpu = malloc_vector_list<int>(n_query, k);
    int** topk_index_gpu = malloc_vector_list<int>(n_query, k);

    /**
     * 将topk距离设为浮点数最小值
     */
    float** topk_dist_cpu = malloc_vector_list<float>(n_query, k);
    float** topk_dist_gpu = malloc_vector_list<float>(n_query, k);
    for(int i = 0; i < n_query; ++i){
        for(int j = 0; j < k; ++j){
            topk_dist_cpu[i][j] = -std::numeric_limits<float>::max();
            topk_dist_gpu[i][j] = -std::numeric_limits<float>::max();
        }
    }

    double gpu_duration_ms = 0, cpu_duration_ms = 0;

    /* GPU计算 - 使用选定的算法 */ 
    MEASURE_MS_AND_SAVE("gpu耗时：", gpu_duration_ms,
        algo_info.func(
            h_query_vectors, h_data_vectors, 
            topk_index_gpu, topk_dist_gpu,
            n_query, n_batch, n_dim, k
        );
    );
    
    /* CPU计算 */ 
    MEASURE_MS_AND_SAVE("cpu耗时：", cpu_duration_ms,
        cpu_cos_distance_topk(
            h_query_vectors, h_data_vectors, 
            topk_index_cpu, topk_dist_cpu,
            n_query, n_batch, n_dim, k
        );
    );

    double speedup = (double)cpu_duration_ms / (double)gpu_duration_ms;

    // 验证结果（只比较距离数组）
    pass &= compare_set_2D<float>(topk_dist_gpu, topk_dist_cpu, n_query, k, EPSILON);
    
    if (!pass) {
        // 可选：打印详细信息用于调试
        // print_2D("cpu topk index", topk_index_cpu, n_query, k); 
        // print_2D("gpu topk index", topk_index_gpu, n_query, k); 
        // print_2D("cpu topk dist", topk_dist_cpu, n_query, k); 
        // print_2D("gpu topk dist", topk_dist_gpu, n_query, k); 
    }    

    // 清理内存
    if (need_free_query) {
        free_vector_list((void**)h_query_vectors);
    }
    if (need_free_data) {
        free_vector_list((void**)h_data_vectors);
    }
    free_vector_list((void**)topk_index_gpu);
    free_vector_list((void**)topk_index_cpu);
    free_vector_list((void**)topk_dist_cpu);
    free_vector_list((void**)topk_dist_gpu);

    return {
        pass ? 1.0 : 0.0,
        (double)n_query, (double)n_batch, (double)n_dim, (double)k, 
        gpu_duration_ms, cpu_duration_ms, speedup, (double)memory_mb
    };

}

// 运行所有算法版本的对比测试
void run_algorithm_comparison(int n_query, int n_batch, int n_dim, int k) {
    
    // 测试所有已注册的算法
    for (auto& pair : algorithm_registry) {
        test_cos_distance_topk_with_algorithm(
            n_query, n_batch, n_dim, k, 
            pair.first
        );
    }   
}

// 解析命令行参数（仅支持 warpsort）
AlgorithmVersion parse_algorithm_version(const char* arg) {
    std::string version(arg);
    if (version == "warpsort" || version == "WARP_SORT" || version == "0") {
        return WARP_SORT;
    } else if (version == "all" || version == "ALL") {
        return ALL_VERSIONS;
    } else {
        COUT_VAL("未知的算法版本: ", arg);
        COUT_VAL("可用选项: warpsort, all");
        exit(1);
    }
}

int main(int argc, char** argv) {
    srand(time(0));
    
    // 初始化算法注册表
    init_algorithm_registry();
    
    bool all_pass = true;

    MetricsCollector metrics;
    metrics.set_columns("pass rate", "n_query", "n_batch", "n_dim", "k", "avg_gpu_ms", "avg_cpu_ms", "avg_speedup", "memory_mb");
    // metrics.set_num_repeats(1);
    
    // 只测试 warpsort 版本
    test_cos_distance_topk_with_algorithm(1024, 128, 512, 16, WARP_SORT); /*warmup*/

    // 缓存的数据集
    const float** cached_h_query_vectors = nullptr;
    const float** cached_h_data_vectors = nullptr;
    
    // 缓存的关键参数
    int cached_n_query = -1;
    int cached_n_batch = -1;
    int cached_n_dim = -1;

    COUT_ENDL("测试算法: 全寄存器 Topk");
    // PARAM_3D(n_query, (8, 32, 128, 512, 2048), 
    //         n_batch, (128, 512, 2048), /* n_batch < k */
    //         n_dim, (512, 1024))   
    PARAM_2D(n_probes, (1, 5, 10, 20, 40), 
            n_batch, (1024)) /* n_batch < k */
    {
        int n_query = 10000;
        int n_dim = 128;
        
        bool need_regenerate_query = (cached_n_query != n_query || 
                                     cached_n_dim != n_dim);
        
        bool need_regenerate_data = (cached_n_batch != n_batch ||
                                    cached_n_dim != n_dim);
        
        if (need_regenerate_query) {
            if (cached_h_query_vectors != nullptr) {
                free_vector_list((void**)cached_h_query_vectors);
                cached_h_query_vectors = nullptr;
            }
            cached_h_query_vectors = generate_vector_list<const float>(n_query, n_dim);
            cached_n_query = n_query;
            cached_n_dim = n_dim;
        }
        
        if (need_regenerate_data) {
            if (cached_h_data_vectors != nullptr) {
                free_vector_list((void**)cached_h_data_vectors);
                cached_h_data_vectors = nullptr;
            }
            cached_h_data_vectors = generate_vector_list<const float>(n_batch, n_dim);
            cached_n_batch = n_batch;
            cached_n_dim = n_dim;
        }
        
        if (!QUIET) {
            if (need_regenerate_query) {
                COUT_ENDL("[INFO] Regenerated query vectors");
            }
            if (need_regenerate_data) {
                COUT_ENDL("[INFO] Regenerated data vectors");
            }
        }
        
        auto avg_result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto result = test_cos_distance_topk_with_algorithm(
                n_query, n_batch, n_dim, n_probes, WARP_SORT,
                cached_h_query_vectors, cached_h_data_vectors
            );
            all_pass &= (result[0] == 1.0);  // 检查 pass 字段
            return result;
        });
    }
    
    // 清理缓存的数据
    if (cached_h_query_vectors != nullptr) {
        free_vector_list((void**)cached_h_query_vectors);
    }
    if (cached_h_data_vectors != nullptr) {
        free_vector_list((void**)cached_h_data_vectors);
    }

    metrics.print_table();

    // 可选：导出为 CSV
    metrics.export_csv("fusion_cos_topk_metrics.csv");
    
    COUT_ENDL("\n所有测试:", all_pass ? "✅ PASS" : "❌ FAIL");
    return 0;
}
