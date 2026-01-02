#include "../../cuda/pch.h"
#include "../common/test_utils.cuh"
#include "../../cuda/warpsortfilter/warpsort_topk.cu"
#include "../cpu_utils/cpu_utils.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <vector>
#include <algorithm>
#include <random>
#include <cfloat>
#include <cmath>
#include <chrono>

using namespace pgvector::warpsort_topk;

#define EPSILON 1e-2f
/**
 * 测试单个参数组合
 */
std::vector<double> test_single_config(int batch_size, int len, int k) {
    // 1. 生成测试数据（使用 generate_vector_list）
    const float** h_input = generate_vector_list<const float>(batch_size, len);
    float** h_output_vals_cpu = malloc_vector_list<float>(batch_size, k);
    int** h_output_idx_cpu = malloc_vector_list<int>(batch_size, k);
    
    // 2. CPU 参考实现（计时）
    double cpu_duration_ms = 0;
    MEASURE_MS_AND_SAVE("CPU select_k", cpu_duration_ms,
        cpu_select_k(h_input, batch_size, len, k, h_output_vals_cpu, h_output_idx_cpu, true);
    );
    
    // 3. 分配设备内存
    float* d_input = nullptr;
    float* d_output_vals = nullptr;
    int* d_output_idx = nullptr;
    
    cudaMalloc(&d_input, batch_size * len * sizeof(float));
    cudaMalloc(&d_output_vals, batch_size * k * sizeof(float));
    cudaMalloc(&d_output_idx, batch_size * k * sizeof(int));
    CHECK_CUDA_ERRORS;
    
    // 4. 复制数据到设备
    cudaMemcpy(d_input, h_input[0], 
               batch_size * len * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;
    
    // 5. GPU kernel 执行（计时）
    double gpu_duration_ms = 0;
    MEASURE_MS_AND_SAVE("GPU select_k", gpu_duration_ms,
        select_k<float, int>(
            d_input, batch_size, len, k,
            d_output_vals, d_output_idx, true, 0
        );
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    );
    
    // 6. 复制结果回主机
    float** h_output_vals_gpu = malloc_vector_list<float>(batch_size, k);
    
    cudaMemcpy(h_output_vals_gpu[0], d_output_vals, 
               batch_size * k * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;
    
    // 7. 验证结果（使用集合比较，因为相同值的顺序可能不同）
    bool all_match = compare_set_2D(h_output_vals_gpu, h_output_vals_cpu, batch_size, k, EPSILON);
    
    // 调试输出（仅对第一个失败的测试）
    static bool debug_printed = false;
    if (!all_match && !debug_printed && batch_size > 0) {
        debug_printed = true;
        printf("\n[DEBUG] First mismatch found. Showing batch 0:\n");
        printf("Input (first %d values): ", std::min(20, len));
        for (int i = 0; i < std::min(20, len); ++i) {
            printf("%.6f ", h_input[0][i]);
        }
        printf("\nCPU output (original): ");
        for (int i = 0; i < k; ++i) {
            printf("%.6f ", h_output_vals_cpu[0][i]);
        }
        printf("\nGPU output (original): ");
        for (int i = 0; i < k; ++i) {
            printf("%.6f ", h_output_vals_gpu[0][i]);
        }
        printf("\nCPU output (sorted): ");
        std::vector<float> cpu_sorted(h_output_vals_cpu[0], h_output_vals_cpu[0] + k);
        std::sort(cpu_sorted.begin(), cpu_sorted.end());
        for (int i = 0; i < k; ++i) {
            printf("%.6f ", cpu_sorted[i]);
        }
        printf("\nGPU output (sorted): ");
        std::vector<float> gpu_sorted(h_output_vals_gpu[0], h_output_vals_gpu[0] + k);
        std::sort(gpu_sorted.begin(), gpu_sorted.end());
        for (int i = 0; i < k; ++i) {
            printf("%.6f ", gpu_sorted[i]);
        }
        printf("\n");
    }
    
    // 8. 清理
    cudaFree(d_input);
    cudaFree(d_output_vals);
    cudaFree(d_output_idx);
    free_vector_list((void**)h_input);
    free_vector_list((void**)h_output_vals_gpu);
    free_vector_list((void**)h_output_vals_cpu);
    free_vector_list((void**)h_output_idx_cpu);
    
    // 9. 计算加速比
    double speedup = (cpu_duration_ms > 0) ? cpu_duration_ms / gpu_duration_ms : 0.0;
    
    // 10. 返回指标向量
    return {
        all_match ? 1.0 : 0.0,  // pass rate
        (double)batch_size,
        (double)len,
        (double)k,
        (double)(batch_size * len),
        gpu_duration_ms,
        cpu_duration_ms,
        speedup
    };
}

/**
 * 主测试函数
 */
int main(int argc, char** argv) {
    srand(time(0));
    
    bool all_pass = true;
    
    MetricsCollector metrics;
    metrics.set_columns("pass rate", "batch_size", "len", "k", 
                       "total_size", "gpu_ms", "cpu_ms", "speedup");
    metrics.set_num_repeats(1);
    
    COUT_ENDL("测试 select_k");
    
    // 使用 PARAM_3D 宏遍历多种参数组合
    PARAM_3D(batch_size, (1, 5, 10, 20, 50, 100, 200, 500, 1000),
             len, (10, 20, 50, 100, 200, 500, 1000),
             k, (5, 10, 16, 32, 64, 128))
    {
        // 只测试 k <= len 的情况
        if (k > len) continue;
        
        auto result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto test_result = test_single_config(batch_size, len, k);
            all_pass &= (test_result[0] == 1.0);
            return test_result;
        });
    }
    
    metrics.print_table();
    
    COUT_ENDL("\n所有测试:", all_pass ? "✅ PASS" : "❌ FAIL");
    
    return all_pass ? 0 : 1;
}
