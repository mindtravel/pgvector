#include <stdlib.h>
#include <limits>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <ctime>

#include "../../cuda/l2norm/l2norm.cuh"
#include "../../cuda/pch.h"
#include "../common/test_utils.cuh"

#define EPSILON 1e-3
// #define EPSILON 1e-4

/**
 * CPU版本的L2范数计算（用于验证）
 */
void cpu_l2_norm(const float* vectors, float* vector_l2_norm, int n_batch, int n_dim) {
    for (int i = 0; i < n_batch; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n_dim; j++) {
            float val = vectors[i * n_dim + j];
            sum += val * val;
        }
        // vector_l2_norm[i] = sum;
        vector_l2_norm[i] = sqrtf(sum);
    }
}

/**
 * 测试函数 - 测试指定版本的L2范数计算
 * @return vector<double>: {pass rate, n_batch, n_dim, gpu_ms, cpu_ms, speedup, memory_mb}
 */
std::vector<double> test_l2_norm(
    int n_batch, 
    int n_dim, 
    L2NormVersion version) 
{
    bool pass = true;
    // if (!QUIET) {
    //     COUT_VAL("配置:" , "n_batch=", n_batch, "n_dim", n_dim);
    // }  
    COUT_VAL("配置:" , "n_batch=", n_batch, "n_dim", n_dim);

    /* 计算内存占用 */
    double memory_mb = (n_batch * n_dim * sizeof(float) + n_batch * sizeof(float)) / (1024.0 * 1024.0);
    
    /* 分配CPU内存 */
    float* h_vectors = (float*)malloc(n_batch * n_dim * sizeof(float));
    float* h_l2_norm_cpu = (float*)malloc(n_batch * sizeof(float));
    float* h_l2_norm_gpu = (float*)malloc(n_batch * sizeof(float));
    
    /* 初始化随机数据 */
    /* 使用较小的随机值范围，避免数值溢出 */
    for (int i = 0; i < n_batch * n_dim; i++) {
        h_vectors[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; /* -1 到 1 之间的随机数 */
    }
    
    /* 确保至少有一些非零向量，避免全部为零向量 */
    if (n_batch > 0 && n_dim > 0) {
        h_vectors[0] = (h_vectors[0] == 0.0f) ? 1.0f : h_vectors[0];
    }
    
    /* 分配GPU内存 */
    float* d_vectors;
    float* d_l2_norm;
    CHECK_CUDA_ERRORS;
    cudaMalloc((void**)&d_vectors, n_batch * n_dim * sizeof(float));
    CHECK_CUDA_ERRORS;
    cudaMalloc((void**)&d_l2_norm, n_batch * sizeof(float));
    CHECK_CUDA_ERRORS;
    
    /* 复制数据到GPU */
    cudaMemcpy(d_vectors, h_vectors, n_batch * n_dim * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;
    
    double gpu_duration_ms = 0, cpu_duration_ms = 0;
    
    /* GPU计算 */
    MEASURE_MS_AND_SAVE("gpu耗时：", gpu_duration_ms,
        compute_l2_norm(d_vectors, d_l2_norm, n_batch, n_dim, version);
        CHECK_CUDA_ERRORS;
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );
    
    /* CPU计算 */
    MEASURE_MS_AND_SAVE("cpu耗时：", cpu_duration_ms,
        cpu_l2_norm(h_vectors, h_l2_norm_cpu, n_batch, n_dim);
    );
    
    /* 从GPU复制结果 */
    cudaMemcpy(h_l2_norm_gpu, d_l2_norm, n_batch * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;
    
    /* 验证结果 */
    pass = compare_1D(h_l2_norm_gpu, h_l2_norm_cpu, n_batch, EPSILON);
    
    if (!pass && !QUIET) {
        COUT_ENDL("结果验证失败！");
        print_1D("CPU结果", h_l2_norm_cpu, n_batch);
        print_1D("GPU结果", h_l2_norm_gpu, n_batch);
    }
    
    double speedup = (cpu_duration_ms > 0) ? (double)cpu_duration_ms / (double)gpu_duration_ms : 0.0;
    
    /* 清理内存 */
    free(h_vectors);
    free(h_l2_norm_cpu);
    free(h_l2_norm_gpu);
    cudaFree(d_vectors);
    cudaFree(d_l2_norm);
    CHECK_CUDA_ERRORS;
    
    return {
        pass ? 1.0 : 0.0,
        (double)n_batch,
        (double)n_dim,
        gpu_duration_ms,
        cpu_duration_ms,
        speedup,
        memory_mb
    };
}

/**
 * 算法版本信息结构
 */
struct L2NormAlgorithmInfo {
    std::string name;
    std::string description;
    L2NormVersion version;
};

/**
 * 算法注册表
 */
std::map<std::string, L2NormAlgorithmInfo> algorithm_registry = {
    {"auto", {"AUTO", "自动选择最佳版本", L2NORM_AUTO}},
    {"basic", {"BASIC", "基础版本：简单共享内存规约", L2NORM_BASIC}},
    {"optimized", {"OPTIMIZED", "优化版本1：根据dim自动选择策略", L2NORM_OPTIMIZED}},
    {"optimized_v2", {"OPTIMIZED_V2", "优化版本2：简化高效版本", L2NORM_OPTIMIZED_V2}},
    {"optimized_v3", {"OPTIMIZED_V3", "优化版本3：float4向量化加载", L2NORM_OPTIMIZED_V3}}
};

/**
 * 解析命令行参数
 */
L2NormVersion parse_version(const char* arg) {
    std::string version_str(arg);
    std::transform(version_str.begin(), version_str.end(), version_str.begin(), ::tolower);
    
    if (algorithm_registry.find(version_str) != algorithm_registry.end()) {
        return algorithm_registry[version_str].version;
    }
    
    /* 尝试数字解析 */
    int version_num = atoi(arg);
    if (version_num >= 0 && version_num <= 4) {
        return (L2NormVersion)version_num;
    }
    
    COUT_VAL("未知的算法版本: ", arg);
    COUT_ENDL("可用选项: auto, basic, optimized, optimized_v2, optimized_v3, all");
    exit(1);
}

int main(int argc, char** argv) {
    srand(time(0));
    
    bool all_pass = true;
    
    MetricsCollector metrics;
    metrics.set_columns("pass rate", "n_batch", "n_dim", "avg_gpu_ms", "avg_cpu_ms", "avg_speedup", "memory_mb");
    metrics.set_num_repeats(3); /* 重复3次取平均值 */
    
    /* 解析命令行参数 */
    bool test_all_versions = false;
    L2NormVersion selected_version = L2NORM_AUTO;
    
    if (argc > 1) {
        std::string arg_str(argv[1]);
        std::transform(arg_str.begin(), arg_str.end(), arg_str.begin(), ::tolower);
        if (arg_str == "all" || arg_str == "compare") {
            test_all_versions = true;
        } else {
            selected_version = parse_version(argv[1]);
        }
    }
    
    /* 如果测试所有版本 */
    if (test_all_versions) {
        for (auto& pair : algorithm_registry) {
            if (pair.first == "auto") continue; /* 跳过auto，因为它是其他版本的组合 */
            
            L2NormVersion version = pair.second.version;
            std::string version_name = pair.second.name;
            std::string version_desc = pair.second.description;
            
            COUT_ENDL("\n========================================");
            COUT_ENDL("测试算法: " + version_name + " - " + version_desc);
            COUT_ENDL("========================================");
            
            MetricsCollector version_metrics;
            version_metrics.set_columns("pass rate", "n_batch", "n_dim", "avg_gpu_ms", "avg_cpu_ms", "avg_speedup", "memory_mb");
            version_metrics.set_num_repeats(3);
            
            /* 使用PARAM_2D宏扫描n_batch和n_dim的组合 */
            PARAM_2D(n_batch, (1024, 4096, 16384), 
                     n_dim, (16, 32, 64, 128, 256, 512, 1024, 2048)) {
                auto avg_result = version_metrics.add_row_averaged([&]() -> std::vector<double> {
                    auto result = test_l2_norm(n_batch, n_dim, version);
                    all_pass &= (result[0] == 1.0);
                    return result;
                });
            }
            
            version_metrics.print_table();
            version_metrics.export_csv("l2norm_" + version_name + "_metrics.csv");
        }
    } else {
        /* 测试单个版本 */
        std::string version_name = "AUTO";
        std::string version_desc = "自动选择";
        for (auto& pair : algorithm_registry) {
            if (pair.second.version == selected_version) {
                version_name = pair.second.name;
                version_desc = pair.second.description;
                break;
            }
        }
        
        COUT_ENDL("\n========================================");
        COUT_ENDL("测试算法: " + version_name + " - " + version_desc);
        COUT_ENDL("========================================");
        
        /* 使用PARAM_2D宏扫描n_batch和n_dim的组合 */
        // PARAM_2D(n_batch, (1,2,3,10,16,50,128,1000), 
        //          n_dim, (10, 16, 20, 32, 64, 114, 128, 200, 256, 333, 512, 1024))
        // PARAM_2D(n_batch, (128, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000),// 6400000, 1280000), 
        PARAM_2D(n_batch, (1000, 1024000),// 6400000, 1280000), 
                n_dim, (512))//, 1024))
                /* 继续增大维度，会因为浮点数的累加出现误差精度的问题，由于项目用不上，所以不再往下探索 */ 
        {
            auto avg_result = metrics.add_row_averaged([&]() -> std::vector<double> {
                auto result = test_l2_norm(n_batch, n_dim, selected_version);
                all_pass &= (result[0] == 1.0);
                return result;
            });
        }
        
        metrics.print_table();
        metrics.export_csv("l2norm_" + version_name + "_metrics.csv");
    }
    
    COUT_ENDL("\n所有测试: " + std::string(all_pass ? "✅ PASS" : "❌ FAIL"));
    return all_pass ? 0 : 1;
}

