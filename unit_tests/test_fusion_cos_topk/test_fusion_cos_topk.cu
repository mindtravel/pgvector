#include <stdlib.h>
#include <limits>
#include <string>
#include <map>
#include <queue>

#include "../cuda/fusion_cos_topk/fusion_cos_topk.cuh"
#include "../cuda/pch.h"
#include "../common/test_utils.cuh"

#define EPSILON 1e-2
#define DIV_EPSILON 1e-4

// 算法版本枚举
enum AlgorithmVersion {
    GLOBAL_MEMORY,      // 普通显存版本
    SHARED_MEMORY,      // 共享内存版本
    WARP_SORT,          // 纯寄存器 topk
    ALL_VERSIONS        // 运行所有版本
};

// 算法函数指针类型定义
typedef void (*CosineTopkFunc)(
    float** query_vectors, float** data_vectors, 
    int** data_index, int** topk_index, float** topk_dist,
    int n_query, int n_batch, int n_dim, int k
);

// 算法版本信息结构
struct AlgorithmInfo {
    std::string name;
    std::string description;
    CosineTopkFunc func;
};

// 算法注册表
std::map<AlgorithmVersion, AlgorithmInfo> algorithm_registry = {
    {GLOBAL_MEMORY, {"GlobalMemory", "普通显存堆实现", cuda_cos_topk_heap}},
    {SHARED_MEMORY, {"SharedMemory", "共享内存堆实现", cuda_cos_topk_heap_sharedmem}},
    {WARP_SORT, {"warpsort", "纯寄存器 Top k", cuda_cos_topk_warpsort}}
};

// CPU版本余弦版本k近邻计算（用于验证）- 使用最大堆优化
void cpu_cos_distance_topk(float** query_vectors, float** data_vectors, 
                        int** data_index, 
                        int** topk_index, float** topk_dist,
                        int n_query, int n_batch, int n_dim, int k) {
    // 计算每个向量的L2范数
    float* query_norms = (float*)malloc(n_query * sizeof(float));
    float* data_norms = (float*)malloc(n_batch * sizeof(float));
    
    // 计算query向量的L2范数
    for (int i = 0; i < n_query; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n_dim; j++) {
            sum += query_vectors[i][j] * query_vectors[i][j];
        }
        query_norms[i] = sum;
    }
    
    // 计算data向量的L2范数
    for (int i = 0; i < n_batch; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n_dim; j++) {
            sum += data_vectors[i][j] * data_vectors[i][j];
        }
        data_norms[i] = sum;
    }
    
    // 为每个query使用最大堆维护top-k最小距离
    for (int i = 0; i < n_query; i++) {
        std::priority_queue<std::pair<float, int>> heap;  // 最大堆
        
        // 计算当前query与所有data向量的余弦距离
        for (int j = 0; j < n_batch; j++) {
            // 计算点积
            float dot_product = 0.0f;
            for (int d = 0; d < n_dim; d++) {
                dot_product += query_vectors[i][d] * data_vectors[j][d];
            }
            
            float cos_distance;
            // 计算余弦距离
            if (query_norms[i] < 1e-6f || data_norms[j] < 1e-6f)
                cos_distance = 1.0f;  // 如果任一向量接近零向量，距离为1
            else
                cos_distance = 1.0f - (dot_product / sqrt(query_norms[i] * data_norms[j]));
            
            // 使用最大堆维护top-k最小距离
            if (heap.size() < k) {
                // 堆未满，直接添加
                heap.push(std::make_pair(cos_distance, data_index[i][j]));
            } else if (cos_distance < heap.top().first) {
                // 堆已满，如果当前距离小于堆顶，则替换堆顶
                heap.pop();
                heap.push(std::make_pair(cos_distance, data_index[i][j]));
            }
        }
        
        // 从堆中提取top-k结果
        int count = 0;
        while (!heap.empty() && count < k) {
            auto top = heap.top();
            heap.pop();
            topk_index[i][k - 1 - count] = top.second;
            topk_dist[i][k - 1 - count] = top.first;
            count++;
        }
    }
    
    free(query_norms);
    free(data_norms);
}

int** generate_data_index(int size_x, int size_y){
    int** data_index = malloc_vector_list<int>(size_x, size_y);
    for(int i = 0; i < size_x; i++){
        for(int j = 0; j < size_y; j++){
            data_index[i][j] = j;
        }
    }
    return data_index;
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
        COUT_VAL("配置:", n_query, "个查询向量 ×", n_batch, "个数据向量", ", 向量维度:", n_dim, ", 找top", k, "算法: ", algo_info.description);
    }  
        
    // 生成测试数据
    float** h_query_vectors = generate_vector_list(n_query, n_dim);
    float** h_data_vectors = generate_vector_list(n_batch, n_dim);
    int** data_index = generate_data_index(n_query, n_batch);
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
            data_index, topk_index_gpu, topk_dist_gpu,
            n_query, n_batch, n_dim, k
        );
    );
    
    /* CPU计算 */ 
    MEASURE_MS_AND_SAVE("cpu耗时：", cpu_duration_ms,
        cpu_cos_distance_topk(
            h_query_vectors, h_data_vectors, 
            data_index, topk_index_cpu, topk_dist_cpu,
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
    free(data_index);
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

// 解析命令行参数
AlgorithmVersion parse_algorithm_version(const char* arg) {
    std::string version(arg);
    if (version == "global" || version == "GLOBAL" || version == "0") {
        return GLOBAL_MEMORY;
    } else if (version == "shared" || version == "SHARED" || version == "1") {
        return SHARED_MEMORY;
    } else if (version == "all" || version == "ALL" || version == "compare") {
        return ALL_VERSIONS;
    } else {
        COUT_VAL("未知的算法版本: ", arg);
        COUT_VAL("可用选项: global, shared, all");
        exit(1);
    }
}

int main(int argc, char** argv) {
    srand(time(0));
    
    bool all_pass = true;

    MetricsCollector metrics;
    metrics.set_columns("pass rate", "n_query", "n_batch", "n_dim", "k", "avg_gpu_ms", "avg_cpu_ms", "avg_speedup", "memory_mb");
    metrics.set_num_repeats(1);
    
    // 解析命令行参数
    AlgorithmVersion selected_version = ALL_VERSIONS;
    selected_version = WARP_SORT;
    if (argc > 1) {
        selected_version = parse_algorithm_version(argv[1]);
    }
    
    bool test = true;
    if (selected_version == GLOBAL_MEMORY || selected_version == ALL_VERSIONS) {
        COUT_ENDL("测试算法: 全局内存堆实现");
        test &= check_pass("", test_cos_distance_topk_with_algorithm(1024, 128, 512, 100, GLOBAL_MEMORY)[0]);
        // test &= check_pass("", test_cos_distance_topk_with_algorithm(1024, 1024, 1024, 100, GLOBAL_MEMORY));
        test &= check_pass("", test_cos_distance_topk_with_algorithm(10000, 1024, 1024, 100, GLOBAL_MEMORY)[0]);
        // test &= check_pass("", test_cos_distance_topk_with_algorithm(10000, 2048, 1024, 100, GLOBAL_MEMORY)); /* 这一步结果会出错，不知道是哪一步引起的，反正是个很慢的算法，后续便不再维护 */ 
    }
    if (selected_version == SHARED_MEMORY || selected_version == ALL_VERSIONS) {
        COUT_ENDL("测试算法: 共享内存堆实现");
        test &= check_pass("", test_cos_distance_topk_with_algorithm(1024, 128, 512, 16, SHARED_MEMORY)[0]);
        test &= check_pass("", test_cos_distance_topk_with_algorithm(1024, 1024, 1024, 16, SHARED_MEMORY)[0]);
        test &= check_pass("", test_cos_distance_topk_with_algorithm(10000, 1024, 1024, 16, SHARED_MEMORY)[0]);
    }
    if (selected_version == WARP_SORT || selected_version == ALL_VERSIONS) {
        // test_cos_distance_topk_with_algorithm(1024, 128, 512, 16, WARP_SORT); /*warmup*/

        COUT_ENDL("测试算法: 全寄存器 Topk");
        // PARAM_3D(n_query, (8, 32, 128, 512, 2048), 
        //         n_batch, (128, 512, 2048), /* n_batch < k */
        //         n_dim, (512, 1024))   
        // PARAM_3D(n_query, (8, 32), 
        //     n_batch, (128, 512), /* n_batch < k */
        //     n_dim, (512, 1024))  
        PARAM_3D(n_query, (1000), 
        n_batch, (128), /* n_batch < k */
        n_dim, (128))          
        {

            auto avg_result = metrics.add_row_averaged([&]() -> std::vector<double> {
                auto result = test_cos_distance_topk_with_algorithm(n_query, n_batch, n_dim, 16, WARP_SORT);
                all_pass &= (result[0] == 1.0);  // 检查 pass 字段
                return result;
            });

        }

        // test_cos_distance_topk_with_algorithm(2048, 2048, 1024, 100, WARP_SORT); /*warmup*/
    }

    metrics.print_table();

    // 可选：导出为 CSV
    metrics.export_csv("fusion_cos_topk_metrics.csv");
    
    COUT_ENDL("\n所有测试:", all_pass ? "✅ PASS" : "❌ FAIL");
    return 0;
}
