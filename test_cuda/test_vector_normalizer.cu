#include "../cuda/normalize.h"
#include "test_utils.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstring>
#include <chrono>
#include <time.h>
#include <stdlib.h>



/*
* 测试辅助函数：作为对比的cpu normalize算法
*/
void normlization_cpu(float** vector_list, int n_batch, int n_dim){
    for(int i=0; i<n_batch; ++i){
        float sum = 0.0;
        for(int j=0; j<n_dim; ++j){
            sum += vector_list[i][j] * vector_list[i][j];
        }  
        sum = sqrt(sum);
        for(int j=0; j<n_dim; ++j){
            vector_list[i][j] /= sum;
        }  
    }
}

/*
用cpu保证每次计算的结果正确
*/
bool test_normlization_cpu(float** vector_list, int n_batch, int n_dim){
    for(int i=0; i<n_batch; ++i){
        float sum = 0.0;
        for(int j=0; j<n_dim; ++j){
            sum += vector_list[i][j] * vector_list[i][j];
        }  
        if(!float_equal(sum, 1.0f)){
            return false;
        }
    }
    return true;
}

/*
测试1：性能测试
测试改变n_batch和n_dim的情况下cpu和gpu的性能
测试n_batch和n_dim超过多少的时候算法会失效
*/ 
void test_performance(int n_batch, int n_dim) {
    std::cout << "=== Test1: Performance ===" << std::endl;
    
    float** vector_list = generate_vector_list(n_batch, n_dim);
    
    std::cout << "nbatch=" << n_batch << " n_dim=" << n_dim << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    normalize(vector_list, n_batch, n_dim);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    assert(test_normlization_cpu(vector_list, n_batch, n_dim));

    std::cout << "gpu耗时：" << duration.count() << "ms" << std::endl;

    vector_list = generate_vector_list(n_batch, n_dim);

    start = std::chrono::high_resolution_clock::now();
    normlization_cpu(vector_list, n_batch, n_dim);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    assert(test_normlization_cpu(vector_list, n_batch, n_dim));

    std::cout << "cpu耗时：" << duration.count() << "ms" << std::endl;
    
    std::cout << "Normalization test passed" << std::endl << std::endl;
}

/*
测试2：零向量处理
零向量normalize后还是零向量
只关注正确性不关注性能
*/ 
void test_zero_vector() {
    // TODO:
}

/*
测试3：压力测试
不断输入很多个向量组，看数据传输的异步优化效果
*/ 
void test_large_scale_async_stress(int n_lists, int n_batch, int n_dim) {
    std::cout << "=== 大规模数据压力测试 ===" << std::endl;
        
    std::cout << "测试规模: " << n_lists << " lists × " 
              << n_batch << " vectors × " 
              << n_dim << " dimensions" << std::endl;
    
    // 计算总内存使用量
    size_t total_memory_mb = (size_t)n_lists * n_batch * n_dim * sizeof(float) / (1024 * 1024);
    std::cout << "总内存使用量: " << total_memory_mb << " MB" << std::endl;
    
    // 生成大规模数据
    float*** vector_lists = generate_large_scale_vectors(n_lists, n_batch, n_dim);
    
    std::cout << "注册内存为页锁定..." << std::endl;
    for (int list_id = 0; list_id < n_lists; list_id++) {
        cudaError_t error = cudaHostRegister(
            vector_lists[list_id][0], 
            n_batch * n_dim * sizeof(float), 
            cudaHostRegisterDefault
        );
        if (error != cudaSuccess) {
            std::cerr << "cudaHostRegister failed for list " << list_id 
                      << ": " << cudaGetErrorString(error) << std::endl;
            return;
        }
    }
    std::cout << "内存注册完成 ✓" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    normalize_async(
        vector_lists, 
        n_lists,
        n_batch, 
        n_dim
    );
            
    // 记录结束时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);   

    // 验证结果
    for (int list_id = 0; list_id < n_lists; list_id++) {
        assert(test_normlization_cpu(vector_lists[list_id], n_batch, n_dim));
    }
        
    // 输出性能统计
    std::cout << "\n=== 性能统计 ===" << std::endl;
    std::cout << "总处理时间: " << duration.count() << " ms" << std::endl;
    std::cout << "平均每list处理时间: " << (float)duration.count() / n_lists << " ms" << std::endl;
    
    // 计算吞吐量
    size_t total_vectors = (size_t)n_lists * n_batch;
    float vectors_per_second = (float)total_vectors / (duration.count() / 1000.0f);
    std::cout << "向量处理吞吐量: " << vectors_per_second << " vectors/second" << std::endl;
    
    // 计算内存带宽
    size_t total_data_processed = total_vectors * n_dim * sizeof(float) * 2; // 读写各一次
    float bandwidth_gbps = (float)total_data_processed / (duration.count() / 1000.0f) / (1024 * 1024 * 1024);
    std::cout << "内存带宽: " << bandwidth_gbps << " GB/s" << std::endl;
    
    // 清理内存
    for (int list_id = 0; list_id < n_lists; list_id++) {
        free(vector_lists[list_id][0]); // 释放连续数据内存
        free(vector_lists[list_id]);    // 释放指针数组
    }
    free(vector_lists);
    
    std::cout << "大规模压力测试完成 ✓" << std::endl;
}

int main() {
    std::cout << "开始VectorNormalizer单元测试..." << std::endl << std::endl;
    
    try {
        test_performance(4096, 1024);
        // test_performance(4096, 1021); n_dim不是2的次幂，规约求和会失效
        // test_performance(8192, 512);
        // test_performance(8192, 256);
        // // test_performance(16184, 256); n_batch 从8192增大到16184肯定会寄
        // test_performance(8192, 128);
        // test_performance(2048, 1024);
        // // test_performance(2048, 2048);n_dim 从1024增大到2048肯定会寄
        // test_performance(8192, 1024);

        test_zero_vector();
        // test_large_scale_async_stress(16, 1024, 512);
        // test_large_scale_async_stress(1024, 1024, 512);
        // test_large_scale_async_stress(10, 10, 512);
        // test_large_scale_async_stress(8192, 1024, 512);

        std::cout << "all test passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
}
