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
    VERSION_NAME1,      // 注释
    VERSION_NAME2,      // 注释
    ALL_VERSIONS,      // 注释
};

// 算法函数指针类型定义
typedef void (*FuncTypename)(
    // 算法参数
);

// 算法版本信息结构
struct AlgorithmInfo {
    std::string name;
    std::string description;
    FuncTypename func;
};

// 算法注册表
std::map<AlgorithmVersion, AlgorithmInfo> algorithm_registry = {
    {VERSION_NAME1, {"version1", "中文解释", funcname1}},
    {VERSION_NAME2, {"version2", "中文解释", funcname2}},
};

// CPU版本余弦版本k近邻计算（用于验证）- 使用最大堆优化
void cpu_func(
    // 参数
) {
    //具体实现
}


/**
 * 测试函数 - 返回所有性能指标
 * @return vector<double>: {pass rate, 指标1, 指标2}
 */
std::vector<double> test_func(
    int n_1, int n_2
    // 参数, AlgorithmVersion algo_version
) 
{
    AlgorithmInfo& algo_info = algorithm_registry[algo_version];
    bool pass = true;
    double memory_mb = // 计算内存占用

    float** data = malloc_vector_list<float>(n_1, n_2);
    float** result_cpu = malloc_vector_list<float>(n_1, n_2);
    float** result_gpu = malloc_vector_list<float>(n_1, n_2);
    
    if (!QUIET) {
        COUT_VAL("配置:" , "参数1", param1);
    }  
        
    // 具体实现

    double gpu_duration_ms = 0, cpu_duration_ms = 0;

    /* GPU计算 - 使用选定的算法 */ 
    MEASURE_MS_AND_SAVE("gpu耗时：", gpu_duration_ms,
        algo_info.func(
            // 参数
        );
    );
    
    /* CPU计算 */ 
    MEASURE_MS_AND_SAVE("cpu耗时：", cpu_duration_ms,
        cpu_func(
            参数
        );
    );

    double speedup = (double)cpu_duration_ms / (double)gpu_duration_ms;

    // 验证结果
    pass &= compare_set_2D(result_gpu, result_cpu, n1, n2);
    
    if (!pass) {
        // 可选：打印详细信息用于调试
        // print_2D("result_cpu", result_cpu, n_1, n_2); 
        // print_2D("result_gpu", result_gpu, n_1, n_2); 
    }    

    // 清理内存
    free_vector_list((void**)data);    

    return {
        pass ? 1.0 : 0.0, // 其他指标和参数
        (double)n_1, (double)n_2,
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
    metrics.set_columns("pass rate", "指标1", "指标2", "avg_gpu_ms", "avg_cpu_ms", "avg_speedup", "memory_mb");
    metrics.set_num_repeats(1);
    
    // 解析命令行参数
    AlgorithmVersion selected_version = ALL_VERSIONS;
    selected_version = VERSION_NAME1;
    if (argc > 1) {
        selected_version = parse_algorithm_version(argv[1]);
    }
    
    bool test = true;
    if (selected_version == VERSION_NAME1 || selected_version == ALL_VERSIONS) {
        COUT_ENDL("测试算法: 版本1");
    }
    if (selected_version == VERSION_NAME2 || selected_version == ALL_VERSIONS) {
        // test_dunc(1024, 128, 512, 16, VERSION_NAME2); /*warmup*/

        COUT_ENDL("测试算法: 全寄存器 Topk");
        PARAM_3D(n_1, (10), 
        n_2, (10), /* n_batch < k */
        n_3, (128))          
        {

            auto avg_result = metrics.add_row_averaged([&]() -> std::vector<double> {
                auto result = test_func(n_1, n_2, n_3, 16, VERSION_NAME2);
                all_pass &= (result[0] == 1.0);  // 检查 pass 字段
                return result;
            });

        }

    }

    metrics.print_table();

    // 可选：导出为 CSV
    metrics.export_csv("fusion_cos_topk_metrics.csv");
    
    COUT_ENDL("\n所有测试:", all_pass ? "✅ PASS" : "❌ FAIL");
    return 0;
}
