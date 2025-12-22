#include <stdlib.h>
#include <limits>
#include <string>
#include <map>
#include <thread>
#include <algorithm>

#include "../cuda/fusion_dist_topk/fusion_dist_topk.cuh"
#include "../cuda/pch.h"
#include "../common/test_utils.cuh"

#define EPSILON 1e-2
#define DIV_EPSILON 1e-4

// 算法版本枚举
enum AlgorithmVersion {
    WARP_SORT,          // 纯寄存器 topk
    ALL_VERSIONS        // 运行所有版本
};

// 算法函数指针类型定义
typedef void (*CosineTopkFunc)(
    float** query_vectors, float** data_vectors, 
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

// CPU版本余弦版本k近邻计算（用于验证）- 多线程版本
void cpu_cos_distance_topk(float** query_vectors, float** data_vectors, 
                        int** topk_index, float** topk_dist,
                        int n_query, int n_batch, int n_dim, int k) {
    // 计算每个向量的L2范数（平方和，用于后续计算）
    float* query_norms = (float*)malloc(n_query * sizeof(float));
    float* data_norms = (float*)malloc(n_batch * sizeof(float));
    
    // 计算query向量的平方和
    compute_squared_sums_batch(query_vectors, query_norms, n_query, n_dim);
    
    // 计算data向量的平方和
    compute_squared_sums_batch(data_vectors, data_norms, n_batch, n_dim);
    
    // 多线程处理：每个线程处理一部分query
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // 默认4个线程
    num_threads = std::min(num_threads, static_cast<unsigned int>(n_query));  // 不超过query数量
    
    // 每个线程处理的工作函数
    auto process_query_range = [&](int start_q, int end_q) {
        // 每个query需要自己的cos_pairs数组
        std::vector<std::pair<float, int>> cos_pairs(n_batch);
        
        for (int i = start_q; i < end_q; i++) {
            // 计算当前query与所有data向量的余弦距离
            for (int j = 0; j < n_batch; j++) {
                // 计算点积
                float dot_product = 0.0f;
                for (int d = 0; d < n_dim; d++) {
                    dot_product += query_vectors[i][d] * data_vectors[j][d];
                }
                float cos_sim;
                // 计算余弦相似度
                if (query_norms[i] < 1e-6f || data_norms[j] < 1e-6f)
                    cos_sim = 0.0f;  // 如果任一向量接近零向量，相似度为0
                else
                    cos_sim = 1.0f - (dot_product / sqrt(query_norms[i] * data_norms[j]));
                
                // 存储余弦相似度和对应的数据索引（直接使用 j）
                cos_pairs[j] = std::make_pair(cos_sim, j);
            }
            
            // 使用partial_sort只排序前k个元素，提高效率
            int topk_count = std::min(k, n_batch);
            std::partial_sort(cos_pairs.begin(), cos_pairs.begin() + topk_count, cos_pairs.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) 
                {
                    if(a.first != b.first){
                        return a.first < b.first;  // 降序排序（距离小的在前）
                    }
                    else{
                        return a.second > b.second;
                    }
                });
            
            // 提取前k个最相似的索引
            for (int j = 0; j < topk_count; ++j) {
                topk_index[i][j] = cos_pairs[j].second;
                topk_dist[i][j] = cos_pairs[j].first;
            }
        }
    };
    
    // 分配任务给各个线程
    std::vector<std::thread> threads;
    int queries_per_thread = (n_query + num_threads - 1) / num_threads;
    
    for (unsigned int t = 0; t < num_threads; t++) {
        int start_q = t * queries_per_thread;
        int end_q = std::min(start_q + queries_per_thread, n_query);
        if (start_q < n_query) {
            threads.emplace_back(process_query_range, start_q, end_q);
        }
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    free(query_norms);
    free(data_norms);
}

/**
 * 测试函数 - 返回所有性能指标
 * @return vector<double>: {pass rate, n_query, n_batch, n_dim, k, gpu_ms, cpu_ms, speedup, memery_mb}
 */
std::vector<double> test_cos_distance_topk_with_algorithm(
    int n_query, int n_batch, int n_dim, int k, 
    AlgorithmVersion algo_version
) 
{
    AlgorithmInfo& algo_info = algorithm_registry[algo_version];
    bool pass = true;
    double memory_mb = (double)(n_query * n_dim + n_batch * n_dim + n_query * n_batch) * sizeof(float) / (double)(1024 * 1024); /* 内存使用量 */
    
    if (!QUIET) {
        COUT_VAL("配置:", n_query, "个查询向量 ×", n_batch, "个数据向量", ", 向量维度:", n_dim, ", 找top", k, ", 算法: ", algo_info.description);
    }  
        
    // 生成测试数据
    float** h_query_vectors = generate_vector_list(n_query, n_dim);
    float** h_data_vectors = generate_vector_list(n_batch, n_dim);
    int** topk_index_cpu = (int**)malloc_vector_list(n_query, k, sizeof(int));
    int** topk_index_gpu = (int**)malloc_vector_list(n_query, k, sizeof(int));

    /**
     * 将topk距离设为浮点数最小值
     */
    float** topk_dist_cpu = (float**)malloc_vector_list(n_query, k, sizeof(float));
    float** topk_dist_gpu = (float**)malloc_vector_list(n_query, k, sizeof(float));
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

    // 验证结果
    // pass &= compare_set_2D(topk_index_gpu, topk_index_cpu, n_query, k);
    pass &= compare_set_2D(topk_dist_gpu, topk_dist_cpu, n_query, k);
    
    if (!pass) {
        // 可选：打印详细信息用于调试
        // print_2D("cpu topk index", topk_index_cpu, n_query, k); 
        // print_2D("gpu topk index", topk_index_gpu, n_query, k); 
        // print_2D("cpu topk dist", topk_dist_cpu, n_query, k); 
        // print_2D("gpu topk dist", topk_dist_gpu, n_query, k); 
    }    

    // 清理内存
    free_vector_list((void**)h_query_vectors);    
    free_vector_list((void**)h_data_vectors);
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

    COUT_ENDL("测试算法: 全寄存器 Topk");
    // PARAM_3D(n_query, (8, 32, 128, 512, 2048), 
    //         n_batch, (128, 512, 2048), /* n_batch < k */
    //         n_dim, (512, 1024))   
    PARAM_2D(n_probes, (1, 5, 10, 20, 40), 
            n_batch, (1024)) /* n_batch < k */
    {
        auto avg_result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto result = test_cos_distance_topk_with_algorithm(10000, n_batch, 128, n_probes, WARP_SORT);
            all_pass &= (result[0] == 1.0);  // 检查 pass 字段
            return result;
        });
    }

    metrics.print_table();

    // 可选：导出为 CSV
    metrics.export_csv("fusion_cos_topk_metrics.csv");
    
    COUT_ENDL("\n所有测试:", all_pass ? "✅ PASS" : "❌ FAIL");
    return 0;
}
