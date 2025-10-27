#include <stdlib.h>
#include <limits>
#include <string>
#include <vector>

#include "stream_pass_data.cuh"
#include "../../cuda/pch.h"
#include "../common/test_utils.cuh"

#define EPSILON 1e-2

// 生成测试数据集 - 复用现有方法
float** generate_test_dataset(int n_groups, int n_vectors, int n_dim) {
    COUT_VAL("生成测试数据集: ", n_groups, "组 × ", n_vectors, "个向量 × ", n_dim, "维");
    
    // 计算总向量数
    int total_vectors = n_groups * n_vectors;
    
    // 使用现有的generate_vector_list方法
    float** dataset = generate_vector_list(total_vectors, n_dim);
    
    return dataset;
}

/**
 * 测试函数 - 返回所有性能指标
 * @return vector<double>: {pass rate, n_groups, n_vectors, n_dim, stream_ms, direct_ms, time_ratio, memory_mb}
 */
std::vector<double> test_stream_vs_direct(
    int n_groups, int n_vectors, int n_dim
) {
    bool pass = true;
    
    // 计算内存使用量 (MB)
    int total_elements = n_groups * n_vectors * n_dim;
    double memory_mb = (double)(total_elements * sizeof(float)) / (1024.0 * 1024.0);
    
    if (!QUIET) {
        COUT_VAL("配置:", n_groups, "组 ×", n_vectors, "个向量", ", 向量维度:", n_dim, ", 内存使用:", memory_mb, "MB");
    }
    
    // 生成测试数据
    float** h_dataset = generate_test_dataset(n_groups, n_vectors, n_dim);
    
    // 分配输出数据内存用于验证
    float* h_stream_output = (float*)malloc(total_elements * sizeof(float));
    float* h_direct_output = (float*)malloc(total_elements * sizeof(float));
    
    if (h_stream_output == NULL || h_direct_output == NULL) {
        fprintf(stderr, "内存分配失败\n");
        free_vector_list((void**)h_dataset);
        return {0.0, (double)n_groups, (double)n_vectors, (double)n_dim, 0.0, 0.0, 0.0, memory_mb};
    }
    
    double stream_duration_ms = 0, direct_duration_ms = 0;
    
    // 流式处理测试
    MEASURE_MS_AND_SAVE("流式处理耗时：", stream_duration_ms,
        stream_pass_data_test(
            h_dataset, n_groups, n_vectors, n_dim, 
            STREAM_MODE, &stream_duration_ms, h_stream_output
        );
    );
    
    // 直接处理测试
    MEASURE_MS_AND_SAVE("直接处理耗时：", direct_duration_ms,
        direct_pass_data_test(
            h_dataset, n_groups, n_vectors, n_dim, 
            &direct_duration_ms, h_direct_output
        );
    );
    
    // 计算时间比例 (直接处理时间 / 流式处理时间)
    double time_ratio = (direct_duration_ms > 0) ? 
        (double)direct_duration_ms / (double)stream_duration_ms : 0.0;
    
    // 验证数据一致性
    bool stream_data_consistent = compare_1D(h_dataset[0], h_stream_output, total_elements, EPSILON);
    bool direct_data_consistent = compare_1D(h_dataset[0], h_direct_output, total_elements, EPSILON);
    
    pass = (stream_duration_ms > 0) && (direct_duration_ms > 0) && 
           stream_data_consistent && direct_data_consistent;
    
    if (!QUIET) {
        COUT_VAL("时间比例 (直接/流式):", time_ratio, "x");
        COUT_VAL("流式处理加速:", (time_ratio > 1.0) ? "是" : "否");
        COUT_VAL("流式数据一致性:", stream_data_consistent ? "✅ 通过" : "❌ 失败");
        COUT_VAL("直接数据一致性:", direct_data_consistent ? "✅ 通过" : "❌ 失败");
    }
    
    // 清理内存
    free_vector_list((void**)h_dataset);
    free(h_stream_output);
    free(h_direct_output);
    
    return {
        pass ? 1.0 : 0.0,
        (double)n_groups, (double)n_vectors, (double)n_dim,
        stream_duration_ms, direct_duration_ms, time_ratio, memory_mb
    };
}

int main(int argc, char** argv) {
    srand(time(0));
    
    bool all_pass = true;
    
    MetricsCollector metrics;
    metrics.set_columns("pass rate", "n_groups", "n_vectors", "n_dim", 
                       "avg_stream_ms", "avg_direct_ms", "avg_time_ratio", "memory_mb");
    
    COUT_ENDL("开始流式vs直接处理性能对比测试...");
    
    // 测试不同规模的数据集
    // 小规模测试
    COUT_ENDL("=== 小规模测试 ===");
    PARAM_3D(n_groups, (3, 10, 50), 
            n_vectors, (10, 50, 100), 
            n_dim, (8, 32, 128))
    {
        auto avg_result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto result = test_stream_vs_direct(n_groups, n_vectors, n_dim);
            all_pass &= (result[0] == 1.0);
            return result;
        });
    }
    
    // 中等规模测试
    COUT_ENDL("=== 中等规模测试 ===");
    PARAM_3D(n_groups, (100, 500), 
            n_vectors, (100, 500), 
            n_dim, (256, 512))
    {
        auto avg_result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto result = test_stream_vs_direct(n_groups, n_vectors, n_dim);
            all_pass &= (result[0] == 1.0);
            return result;
        });
    }
    
    // 大规模测试
    COUT_ENDL("=== 大规模测试 ===");
    PARAM_3D(n_groups, (1000), 
            n_vectors, (1000), 
            n_dim, (1024))
    {
        auto avg_result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto result = test_stream_vs_direct(n_groups, n_vectors, n_dim);
            all_pass &= (result[0] == 1.0);
            return result;
        });
    }
    
    metrics.print_table();
    
    // 导出CSV结果
    metrics.export_csv("stream_pass_data_metrics.csv");
    
    COUT_ENDL("\n所有测试:", all_pass ? "✅ PASS" : "❌ FAIL");
    
    if (all_pass) {
        COUT_ENDL("流式处理性能测试完成！");
        COUT_ENDL("注意：空核函数仅用于测试数据传输性能");
        COUT_ENDL("后续可以替换empty_kernel为实际的业务逻辑");
    }
    
    return 0;
}
