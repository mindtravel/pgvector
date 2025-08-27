#include "../cuda/distances.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// 测试辅助函数：比较浮点数
bool float_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// CPU版本的L2距离计算（用于验证）
float cpu_l2_distance(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

// 测试1：基本L2距离计算
void test_basic_l2_distance() {
    std::cout << "=== 测试1：基本L2距离计算 ===" << std::endl;
    
    const int n = 4;
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    
    std::cout << "向量A: ";
    for (int i = 0; i < n; i++) std::cout << a[i] << " ";
    std::cout << std::endl;
    
    std::cout << "向量B: ";
    for (int i = 0; i < n; i++) std::cout << b[i] << " ";
    std::cout << std::endl;
    
    // 分配GPU内存
    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    L2DistanceOp l2_op;
    float gpu_distance = l2_op(d_a, d_b, n);
    float cpu_distance = cpu_l2_distance(a, b, n);
    
    std::cout << "GPU L2距离: " << gpu_distance << std::endl;
    std::cout << "CPU L2距离: " << cpu_distance << std::endl;
    
    assert(float_equal(gpu_distance, cpu_distance));
    
    // 清理GPU内存
    cudaFree(d_a);
    cudaFree(d_b);
    
    std::cout << "✓ 基本L2距离测试通过" << std::endl << std::endl;
}

// 测试2：相同向量
void test_same_vectors() {
    std::cout << "=== 测试2：相同向量 ===" << std::endl;
    
    const int n = 4;
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    std::cout << "向量A: ";
    for (int i = 0; i < n; i++) std::cout << a[i] << " ";
    std::cout << std::endl;
    
    std::cout << "向量B: ";
    for (int i = 0; i < n; i++) std::cout << b[i] << " ";
    std::cout << std::endl;
    
    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    L2DistanceOp l2_op;
    float gpu_distance = l2_op(d_a, d_b, n);
    float cpu_distance = cpu_l2_distance(a, b, n);
    
    std::cout << "GPU L2距离: " << gpu_distance << std::endl;
    std::cout << "CPU L2距离: " << cpu_distance << std::endl;
    
    assert(float_equal(gpu_distance, cpu_distance));
    assert(float_equal(gpu_distance, 0.0f)); // 相同向量距离为0
    
    cudaFree(d_a);
    cudaFree(d_b);
    
    std::cout << "✓ 相同向量测试通过" << std::endl << std::endl;
}

// 测试3：单位向量
void test_unit_vectors() {
    std::cout << "=== 测试3：单位向量 ===" << std::endl;
    
    const int n = 4;
    float a[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f, 0.0f};
    
    std::cout << "向量A: ";
    for (int i = 0; i < n; i++) std::cout << a[i] << " ";
    std::cout << std::endl;
    
    std::cout << "向量B: ";
    for (int i = 0; i < n; i++) std::cout << b[i] << " ";
    std::cout << std::endl;
    
    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    L2DistanceOp l2_op;
    float gpu_distance = l2_op(d_a, d_b, n);
    float cpu_distance = cpu_l2_distance(a, b, n);
    
    std::cout << "GPU L2距离: " << gpu_distance << std::endl;
    std::cout << "CPU L2距离: " << cpu_distance << std::endl;
    
    assert(float_equal(gpu_distance, cpu_distance));
    assert(float_equal(gpu_distance, sqrtf(2.0f))); // 正交单位向量距离为√2
    
    cudaFree(d_a);
    cudaFree(d_b);
    
    std::cout << "✓ 单位向量测试通过" << std::endl << std::endl;
}

// 测试4：零向量
void test_zero_vectors() {
    std::cout << "=== 测试4：零向量 ===" << std::endl;
    
    const int n = 4;
    float a[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float b[] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    std::cout << "向量A (零向量): ";
    for (int i = 0; i < n; i++) std::cout << a[i] << " ";
    std::cout << std::endl;
    
    std::cout << "向量B: ";
    for (int i = 0; i < n; i++) std::cout << b[i] << " ";
    std::cout << std::endl;
    
    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    L2DistanceOp l2_op;
    float gpu_distance = l2_op(d_a, d_b, n);
    float cpu_distance = cpu_l2_distance(a, b, n);
    
    std::cout << "GPU L2距离: " << gpu_distance << std::endl;
    std::cout << "CPU L2距离: " << cpu_distance << std::endl;
    
    assert(float_equal(gpu_distance, cpu_distance));
    assert(float_equal(gpu_distance, sqrtf(30.0f))); // √(1²+2²+3²+4²) = √30
    
    cudaFree(d_a);
    cudaFree(d_b);
    
    std::cout << "✓ 零向量测试通过" << std::endl << std::endl;
}

// 测试5：大维度向量
void test_large_vectors() {
    std::cout << "=== 测试5：大维度向量 ===" << std::endl;
    
    const int n = 1024;
    float* a = new float[n];
    float* b = new float[n];
    
    // 初始化向量
    for (int i = 0; i < n; i++) {
        a[i] = (float)(i + 1);
        b[i] = (float)(i + 2);
    }
    
    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    L2DistanceOp l2_op;
    float gpu_distance = l2_op(d_a, d_b, n);
    float cpu_distance = cpu_l2_distance(a, b, n);
    
    std::cout << "GPU L2距离: " << gpu_distance << std::endl;
    std::cout << "CPU L2距离: " << cpu_distance << std::endl;
    
    assert(float_equal(gpu_distance, cpu_distance, 1e-4f)); // 大维度允许稍大的误差
    
    cudaFree(d_a);
    cudaFree(d_b);
    delete[] a;
    delete[] b;
    
    std::cout << "✓ 大维度向量测试通过" << std::endl << std::endl;
}

// 测试6：性能测试
void test_performance() {
    std::cout << "=== 测试6：性能测试 ===" << std::endl;
    
    const int n = 1024;
    const int iterations = 1000;
    float* a = new float[n];
    float* b = new float[n];
    
    // 初始化向量
    for (int i = 0; i < n; i++) {
        a[i] = (float)(i + 1);
        b[i] = (float)(i + 2);
    }
    
    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    L2DistanceOp l2_op;
    
    // 预热
    for (int i = 0; i < 10; i++) {
        l2_op(d_a, d_b, n);
    }
    
    // GPU性能测试
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        l2_op(d_a, d_b, n);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // CPU性能测试
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        cpu_l2_distance(a, b, n);
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "GPU执行 " << iterations << " 次耗时: " << gpu_duration.count() << "ms" << std::endl;
    std::cout << "CPU执行 " << iterations << " 次耗时: " << cpu_duration.count() << "ms" << std::endl;
    std::cout << "GPU加速比: " << (float)cpu_duration.count() / gpu_duration.count() << "x" << std::endl;
    
    cudaFree(d_a);
    cudaFree(d_b);
    delete[] a;
    delete[] b;
    
    std::cout << "✓ 性能测试完成" << std::endl << std::endl;
}

int main() {
    std::cout << "开始L2DistanceOp单元测试..." << std::endl << std::endl;
    
    try {
        test_basic_l2_distance();
        test_same_vectors();
        test_unit_vectors();
        test_zero_vectors();
        test_large_vectors();
        test_performance();
        
        std::cout << "🎉 所有L2DistanceOp测试通过！" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
}
