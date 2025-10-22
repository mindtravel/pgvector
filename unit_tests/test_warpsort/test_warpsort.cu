/**
 * Unit tests for Warp-Sort Top-K implementation
 * 
 * This file tests the warp-sort based top-k selection algorithm
 * implemented in wrapsort.cu, which is based on RAFT/cuVS warpsort.
 */

#include "pch.h"
#include "../common/test_utils.cuh"

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

#define EPSILON 1e-5f

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * CPU reference implementation for top-k selection
 */
template<typename T, typename IdxT>
void cpu_select_k(
    const T** input,
    int batch_size,
    int len,
    int k,
    T** output_vals,
    IdxT** output_idx,
    bool select_min)
{
    if (k > len) {
        k = len;
    }

    for(int b = 0; b < batch_size; ++b){
        std::vector<std::pair<T, IdxT>> pairs;
        pairs.reserve(len);
        
        for (int i = 0; i < len; i++) {
            pairs.push_back({input[b][i], static_cast<IdxT>(i)});
        }
        
        // 使用 nth_element 进行部分排序
        if (select_min) {
            // 找出最小的 k 个元素，放在 pairs 的前 k 个位置
            std::nth_element(
                pairs.begin(),
                pairs.begin() + k - 1, // 指向第 k 小元素的迭代器
                pairs.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; }
            );
            // 如果需要，对前 k 个元素进行排序（例如需要按升序输出）
            std::sort(
                pairs.begin(),
                pairs.begin() + k,
                [](const auto& a, const auto& b) { return a.first < b.first; }
            );
        } else {
            // 找出最大的 k 个元素，放在 pairs 的前 k 个位置
            std::nth_element(
                pairs.begin(),
                pairs.begin() + k - 1, // 指向第 k 大元素的迭代器
                pairs.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; }
            );
            // 如果需要，对前 k 个元素进行排序（例如需要按降序输出）
            std::sort(
                pairs.begin(),
                pairs.begin() + k,
                [](const auto& a, const auto& b) { return a.first > b.first; }
            );
        }
        
        // Copy results
        for (int i = 0; i < k; i++) {
            output_vals[b][i] = pairs[i].first;
            output_idx[b][i] = pairs[i].second;
        }        
    }
}

/**
 * 测试函数 - 返回所有性能指标
 * @return vector<double>: {pass rate, batch_size, k, len, gpu_ms, cpu_ms, speedup}
 */
std::vector<double> test_warpsort(
    int batch_size, int len, int k
)
{    
    bool pass = true;

    if (!QUIET) {
        COUT_ENDL("配置: ", "batch_size=", batch_size, ", len=", len, ", k=",k);
    }

    // Allocate host memory & generate random data
    srand(42);
    const float** h_input = const_cast<const float**>((float**)generate_vector_list(batch_size, len));
    float** h_gpu_vals = (float**)malloc_vector_list(batch_size, k, sizeof(float));
    float** h_cpu_vals = (float**)malloc_vector_list(batch_size, k, sizeof(float));
    int** h_gpu_idx = (int**)malloc_vector_list(batch_size, k, sizeof(int));
    int** h_cpu_idx = (int**)malloc_vector_list(batch_size, k, sizeof(int));

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
        cpu_select_k<float, int>(
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
    free_vector_list((void**)h_input);
    free_vector_list((void**)h_gpu_vals);
    free_vector_list((void**)h_gpu_idx);
    free_vector_list((void**)h_cpu_vals);
    free_vector_list((void**)h_cpu_idx);
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
        float** h_input = generate_vector_list(1, batch_size * len);
        
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
        
        free_vector_list((void**)h_input);
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
        
    PARAM_2D(batch, (2000, 5000, 10000, 20000), 
                k, (8, 16, 32, 50, 64, 100, 128))        
    // PARAM_2D(batch, (100, 200), 
    //             k, (8, 16))        
    {
        
        auto avg_result = metrics.add_row_averaged([&]() -> std::vector<double> {
            auto result = test_warpsort(batch, 1024, k);
            all_pass &= (result[0] == 1.0);  // 检查 pass 字段
            return result;
        });
        
    }
    
    metrics.print_table();
    
    // 可选：导出为 CSV
    metrics.export_csv("warpsort_metrics.csv");
    
    COUT_ENDL("\n所有测试:", all_pass ? "✅ PASS" : "❌ FAIL");
    return 0;
}
