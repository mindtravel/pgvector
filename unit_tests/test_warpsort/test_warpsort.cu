/**
 * Unit tests for Warp-Sort Top-K implementation
 * 
 * This file tests the warp-sort based top-k selection algorithm
 * implemented in wrapsort.cu, which is based on RAFT/cuVS warpsort.
 */

#include "pch.h"
#include "../common/test_utils.cuh"
#include "../cpu_utils/cpu_utils.h"

// Forward declarations from warpsortfilter/wrapsort_topk.cu
namespace pgvector {
namespace warpsort_topk {

template<typename T, typename IdxT>
cudaError_t select_k(
    const T* input,
    int batch_size,
    int len,
    int k,
    T* output_vals,
    IdxT* output_idx,
    bool select_min,
    cudaStream_t stream);

}
}

/**
 * 测试函数 - 返回所有性能指标
 * @param batch_size 批次大小
 * @param len 每个向量的长度
 * @param k top-k值
 * @param h_input 预生成的输入数据（如果为nullptr，则内部生成）
 * @return vector<double>: {pass rate, batch_size, k, len, gpu_ms, cpu_ms, speedup}
 */
std::vector<double> test_warpsort(
    int batch_size, int len, int k,
    const float** h_input = nullptr
)
{    
    bool pass = true;

    if (!QUIET) {
        COUT_ENDL("配置: ", "batch_size=", batch_size, ", len=", len, ", k=",k);
    }

    // Allocate host memory & generate random data (if not provided)
    bool need_free_input = false;
    if (h_input == nullptr) {
        srand(42);
        h_input = generate_vector_list<const float>(batch_size, len);
        need_free_input = true;
    }
    float** h_gpu_vals = malloc_vector_list<float>(batch_size, k);
    float** h_cpu_vals = malloc_vector_list<float>(batch_size, k);
    int** h_gpu_idx = malloc_vector_list<int>(batch_size, k);
    int** h_cpu_idx = malloc_vector_list<int>(batch_size, k);

    // Allocate device memory
    float *d_input, *d_output_vals;
    int *d_output_idx;
    cudaMalloc(&d_input, batch_size * len * sizeof(float));
    cudaMalloc(&d_output_vals, batch_size * k * sizeof(float));
    cudaMalloc(&d_output_idx, batch_size * k * sizeof(int));
    CHECK_CUDA_ERRORS;

    double gpu_duration_ms = 0, cpu_duration_ms = 0;

    // Run GPU kernel
    MEASURE_MS_AND_SAVE("gpu耗时：", gpu_duration_ms,
        // Copy input to device
        cudaMemcpy(d_input, h_input[0], batch_size * len * sizeof(float), 
                cudaMemcpyHostToDevice);
        CHECK_CUDA_ERRORS;

        cudaError_t err = pgvector::warpsort_topk::select_k<float, int>(
            d_input, batch_size, len, k, d_output_vals, d_output_idx, true, 0);
        if (err != cudaSuccess) {
            COUT_ENDL("  ❌ GPU kernel failed:", cudaGetErrorString(err));
        }
        cudaMemcpy(h_gpu_vals[0], d_output_vals, batch_size * k * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_gpu_idx[0], d_output_idx, batch_size * k * sizeof(int), 
                   cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS;
    );


    // Run CPU select with timing
    MEASURE_MS_AND_SAVE("cpu耗时：", cpu_duration_ms,
        cpu_select_k(
            h_input, batch_size, len, k, h_cpu_vals, h_cpu_idx, true
        );
    );

    double speedup = (double)cpu_duration_ms / (double)gpu_duration_ms;

    // pass &= count_equal_elements_set_2D(h_gpu_idx, h_cpu_idx, batch_size, k);
    pass &= count_equal_elements_set_2D(h_gpu_vals, h_cpu_vals, batch_size, k);

    if (!pass && !QUIET) {
        // 可选：打印详细信息用于调试
        // print_2D("cpu topk index", topk_index_cpu, n_query, k); 
        // print_2D("gpu topk index", topk_index_gpu, n_query, k); 
        // print_2D("cpu topk dist", topk_dist_cpu, n_query, k); 
        // print_2D("gpu topk dist", topk_dist_gpu, n_query, k); 
    }
    
    // Cleanup
    if (need_free_input) {
        free_vector_list(h_input);
    }
    free_vector_list(h_gpu_vals);
    free_vector_list(h_gpu_idx);
    free_vector_list(h_cpu_vals);
    free_vector_list(h_cpu_idx);
    cudaFree(d_input);
    cudaFree(d_output_vals);
    cudaFree(d_output_idx);
    
    /* 返回所有指标 */
    return {
        pass ? 1.0 : 0.0,
        (double)batch_size, (double)k, (double)len, 
        gpu_duration_ms, cpu_duration_ms, speedup 
    };
}

/**
 * Test 5: Performance benchmark
 */
void test_performance()
{
    COUT_ENDL("\n=== Test 5: 性能基准测试 ===");
    
    const int batch_size = 100;
    const int len = 10000;
    
    std::vector<int> k_values = {16, 32, 64, 128, 256};
    
    COUT_TABLE("k", "avg_time(ms)", "throughput(q/s)");
    
    for (int k : k_values) {
        const float** h_input = generate_vector_list<const float>(1, batch_size * len);
        
        float *d_input, *d_output_vals;
        int *d_output_idx;
        cudaMalloc(&d_input, batch_size * len * sizeof(float));
        cudaMalloc(&d_output_vals, batch_size * k * sizeof(float));
        cudaMalloc(&d_output_idx, batch_size * k * sizeof(int));
        CHECK_CUDA_ERRORS;
        
        cudaMemcpy(d_input, h_input[0], batch_size * len * sizeof(float), 
                   cudaMemcpyHostToDevice);
        CHECK_CUDA_ERRORS;
        
        // Warmup
        pgvector::warpsort_topk::select_k<float, int>(
            d_input, batch_size, len, k, d_output_vals, d_output_idx, true, 0);
        cudaDeviceSynchronize();
        
        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        const int n_iterations = 100;
        cudaEventRecord(start);
        for (int i = 0; i < n_iterations; i++) {
            pgvector::warpsort_topk::select_k<float, int>(
                d_input, batch_size, len, k, d_output_vals, d_output_idx, true, 0);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        float avg_ms = ms / n_iterations;
        float throughput = batch_size / avg_ms * 1000.0f;
        
        COUT_TABLE(k, avg_ms, (int)throughput);
        
        free_vector_list(h_input);
        cudaFree(d_input);
        cudaFree(d_output_vals);
        cudaFree(d_output_idx);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

int main()
{    
    test_warpsort(1000, 1024, 128); /* warm up */

    COUT_ENDL("\n=== Test 1: 性能测试 ===");
    
    bool all_pass = true;
    
    MetricsCollector metrics;
    metrics.set_columns("pass rate", "batch", "len", "k", "avg_gpu_ms", "avg_cpu_ms", "avg_speedup");
    // metrics.set_num_repeats(1);
    
    // 缓存的数据集
    const float** cached_h_input = nullptr;
    
    // 缓存的关键参数
    int cached_batch_size = -1;
    int cached_len = -1;
        
    PARAM_2D(batch, (2000, 5000, 10000, 20000), 
                k, (8, 16, 32, 50, 64, 100, 128))        
    // PARAM_2D(batch, (100, 200), 
    //             k, (8, 16))        
    {
        int len = 1024;  // 固定长度
        
        bool need_regenerate_input = (cached_batch_size != batch || 
                                     cached_len != len);
        
        if (need_regenerate_input) {
            if (cached_h_input != nullptr) {
                free_vector_list(cached_h_input);
                cached_h_input = nullptr;
            }
            srand(42);
            cached_h_input = generate_vector_list<const float>(batch, len);
            cached_batch_size = batch;
            cached_len = len;
        }
        
        if (!QUIET) {
            if (need_regenerate_input) {
                COUT_ENDL("[INFO] Regenerated input dataset");
            }
        }
        
        auto avg_result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto result = test_warpsort(batch, len, k, cached_h_input);
            all_pass &= (result[0] == 1.0);  // 检查 pass 字段
            return result;
        });
        
    }
    
    // 清理缓存的数据
    if (cached_h_input != nullptr) {
        free_vector_list(cached_h_input);
    }
    
    metrics.print_table();
    
    // 可选：导出为 CSV
    metrics.export_csv("warpsort_metrics.csv");
    
    COUT_ENDL("\n所有测试:", all_pass ? "✅ PASS" : "❌ FAIL");
    return 0;
}
