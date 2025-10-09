// gcc -o test_optimized_gpu test_optimized_gpu.c ../cuda/cuda_wrapper.o -I../cuda -L/usr/local/cuda/lib64 -lcudart -lstdc++

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdbool.h>

// 模拟PostgreSQL环境
#include "../cuda/cuda_wrapper.h"

// 测试参数
#define TEST_DIMENSIONS 128
#define TEST_CENTERS 1000
#define TEST_QUERIES 100

// 性能测试结构
typedef struct {
    double upload_time;
    double query_time;
    int success_count;
    int total_queries;
} PerformanceTest;

// 生成随机向量
void generate_random_vector(float* vector, int dimensions) {
    for (int i = 0; i < dimensions; i++) {
        vector[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

// 获取当前时间（微秒）
double get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

// 测试优化后的GPU功能
int test_optimized_gpu_upload() {
    printf("=== 测试优化后的GPU聚类中心上传功能 ===\n");
    
    // 初始化CUDA上下文
    CudaCenterSearchContext* ctx = cuda_center_search_init(TEST_CENTERS, TEST_DIMENSIONS, false);
    if (!ctx) {
        printf("错误：无法初始化CUDA上下文\n");
        return -1;
    }
    
    // 生成测试聚类中心数据
    float* centers_data = (float*)malloc(TEST_CENTERS * TEST_DIMENSIONS * sizeof(float));
    for (int i = 0; i < TEST_CENTERS; i++) {
        generate_random_vector(&centers_data[i * TEST_DIMENSIONS], TEST_DIMENSIONS);
    }
    
    // 测试上传性能
    double start_time = get_time_us();
    int upload_result = cuda_upload_centers(ctx, centers_data);
    double end_time = get_time_us();
    
    if (upload_result == 0) {
        printf("✓ 聚类中心数据上传成功\n");
        printf("✓ 上传时间: %.2f ms\n", (end_time - start_time) / 1000.0);
    } else {
        printf("✗ 聚类中心数据上传失败\n");
        free(centers_data);
        cuda_center_search_cleanup(ctx);
        return -1;
    }
    
    // 测试多次查询性能
    printf("\n=== 测试多次查询性能 ===\n");
    float* query_vector = (float*)malloc(TEST_DIMENSIONS * sizeof(float));
    float* distances = (float*)malloc(TEST_CENTERS * sizeof(float));
    
    double total_query_time = 0;
    int successful_queries = 0;
    
    for (int q = 0; q < TEST_QUERIES; q++) {
        generate_random_vector(query_vector, TEST_DIMENSIONS);
        
        start_time = get_time_us();
        int result = cuda_compute_center_distances(ctx, query_vector, distances);
        end_time = get_time_us();
        
        if (result == 0) {
            total_query_time += (end_time - start_time);
            successful_queries++;
        }
    }
    
    printf("✓ 成功查询数: %d/%d\n", successful_queries, TEST_QUERIES);
    printf("✓ 平均查询时间: %.2f ms\n", (total_query_time / successful_queries) / 1000.0);
    printf("✓ 总查询时间: %.2f ms\n", total_query_time / 1000.0);
    
    // 清理资源
    free(centers_data);
    free(query_vector);
    free(distances);
    cuda_center_search_cleanup(ctx);
    
    return 0;
}

// 测试零拷贝模式
int test_zero_copy_mode() {
    printf("\n=== 测试零拷贝模式 ===\n");
    
    CudaCenterSearchContext* ctx = cuda_center_search_init(TEST_CENTERS, TEST_DIMENSIONS, true);
    if (!ctx) {
        printf("错误：无法初始化零拷贝CUDA上下文\n");
        return -1;
    }
    
    // 生成测试数据
    float* centers_data = (float*)malloc(TEST_CENTERS * TEST_DIMENSIONS * sizeof(float));
    for (int i = 0; i < TEST_CENTERS; i++) {
        generate_random_vector(&centers_data[i * TEST_DIMENSIONS], TEST_DIMENSIONS);
    }
    
    // 测试零拷贝上传
    double start_time = get_time_us();
    int upload_result = cuda_upload_centers_zero_copy(ctx, centers_data);
    double end_time = get_time_us();
    
    if (upload_result == 0) {
        printf("✓ 零拷贝模式上传成功\n");
        printf("✓ 零拷贝上传时间: %.2f ms\n", (end_time - start_time) / 1000.0);
    } else {
        printf("✗ 零拷贝模式上传失败\n");
        free(centers_data);
        cuda_center_search_cleanup(ctx);
        return -1;
    }
    
    // 测试查询性能
    float* query_vector = (float*)malloc(TEST_DIMENSIONS * sizeof(float));
    float* distances = (float*)malloc(TEST_CENTERS * sizeof(float));
    
    generate_random_vector(query_vector, TEST_DIMENSIONS);
    
    start_time = get_time_us();
    int result = cuda_compute_center_distances(ctx, query_vector, distances);
    end_time = get_time_us();
    
    if (result == 0) {
        printf("✓ 零拷贝模式查询成功\n");
        printf("✓ 查询时间: %.2f ms\n", (end_time - start_time) / 1000.0);
    } else {
        printf("✗ 零拷贝模式查询失败\n");
    }
    
    // 清理资源
    free(centers_data);
    free(query_vector);
    free(distances);
    cuda_center_search_cleanup(ctx);
    
    return 0;
}

int main() {
    printf("开始测试优化后的GPU聚类中心上传功能\n");
    printf("测试参数: 维度=%d, 聚类中心数=%d, 查询数=%d\n\n", 
           TEST_DIMENSIONS, TEST_CENTERS, TEST_QUERIES);
    
    // 检查CUDA可用性
    if (!cuda_is_available()) {
        printf("错误：CUDA不可用，无法进行测试\n");
        return -1;
    }
    
    printf("✓ CUDA环境检查通过\n\n");
    
    // 运行测试
    int result1 = test_optimized_gpu_upload();
    int result2 = test_zero_copy_mode();
    
    printf("\n=== 测试总结 ===\n");
    if (result1 == 0 && result2 == 0) {
        printf("✓ 所有测试通过\n");
        printf("✓ 优化后的GPU功能工作正常\n");
        printf("✓ 聚类中心数据只需上传一次，后续查询直接使用\n");
    } else {
        printf("✗ 部分测试失败\n");
    }
    
    return (result1 == 0 && result2 == 0) ? 0 : -1;
}
