#include "../cuda/distances.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

// 测试辅助函数：比较浮点数
bool float_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// CPU版本的余弦距离计算（用于验证）
float cpu_cosine_distance(const float* a, const float* b, int n) {
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    
    for (int i = 0; i < n; i++) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);
    
    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 1.0f; // 如果任一向量为零向量，距离为1
    }
    
    float similarity = dot_product / (norm_a * norm_b);
    return 1.0f - similarity;
}

// 测试1：基本余弦距离计算
void test_basic_cosine_distance() {
    std::cout << "=== 测试1：基本余弦距离计算 ===" << std::endl;
    
    const int n = 4;
    float a[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f, 0.0f};
    
    std::cout << "向量A: ";
    for (int i = 0; i < n; i++) std::cout << a[i] << " ";
    std::cout << std::endl;
    
    std::cout << "向量B: ";
    for (int i = 0; i < n; i++) std::cout << b[i] << " ";
    std::cout << std::endl;
    
    CosineDistanceOp cosine_op(n);
    float gpu_distance = cosine_op.compute(a, b);
    float cpu_distance = cpu_cosine_distance(a, b, n);
    
    std::cout << "GPU余弦距离: " << gpu_distance << std::endl;
    std::cout << "CPU余弦距离: " << cpu_distance << std::endl;
    
    assert(float_equal(gpu_distance, cpu_distance));
    assert(float_equal(gpu_distance, 1.0f)); // 正交向量距离为1
    
    std::cout << "✓ 基本余弦距离测试通过" << std::endl << std::endl;
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
    
    CosineDistanceOp cosine_op(n);
    float gpu_distance = cosine_op.compute(a, b);
    float cpu_distance = cpu_cosine_distance(a, b, n);
    
    std::cout << "GPU余弦距离: " << gpu_distance << std::endl;
    std::cout << "CPU余弦距离: " << cpu_distance << std::endl;
    
    assert(float_equal(gpu_distance, cpu_distance));
    assert(float_equal(gpu_distance, 0.0f)); // 相同向量距离为0
    
    std::cout << "✓ 相同向量测试通过" << std::endl << std::endl;
}

// 测试3：相反向量
void test_opposite_vectors() {
    std::cout << "=== 测试3：相反向量 ===" << std::endl;
    
    const int n = 4;
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {-1.0f, -2.0f, -3.0f, -4.0f};
    
    std::cout << "向量A: ";
    for (int i = 0; i < n; i++) std::cout << a[i] << " ";
    std::cout << std::endl;
    
    std::cout << "向量B: ";
    for (int i = 0; i < n; i++) std::cout << b[i] << " ";
    std::cout << std::endl;
    
    CosineDistanceOp cosine_op(n);
    float gpu_distance = cosine_op.compute(a, b);
    float cpu_distance = cpu_cosine_distance(a, b, n);
    
    std::cout << "GPU余弦距离: " << gpu_distance << std::endl;
    std::cout << "CPU余弦距离: " << cpu_distance << std::endl;
    
    assert(float_equal(gpu_distance, cpu_distance));
    assert(float_equal(gpu_distance, 2.0f)); // 相反向量距离为2
    
    std::cout << "✓ 相反向量测试通过" << std::endl << std::endl;
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
    
    CosineDistanceOp cosine_op(n);
    float gpu_distance = cosine_op.compute(a, b);
    float cpu_distance = cpu_cosine_distance(a, b, n);
    
    std::cout << "GPU余弦距离: " << gpu_distance << std::endl;
    std::cout << "CPU余弦距离: " << cpu_distance << std::endl;
    
    assert(float_equal(gpu_distance, cpu_distance));
    assert(float_equal(gpu_distance, 1.0f)); // 零向量与任何非零向量距离为1
    
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
        b[i] = (float)(n - i);
    }
    
    CosineDistanceOp cosine_op(n);
    float gpu_distance = cosine_op.compute(a, b);
    float cpu_distance = cpu_cosine_distance(a, b, n);
    
    std::cout << "GPU余弦距离: " << gpu_distance << std::endl;
    std::cout << "CPU余弦距离: " << cpu_distance << std::endl;
    
    assert(float_equal(gpu_distance, cpu_distance, 1e-4f)); // 大维度允许稍大的误差
    
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
    
    CosineDistanceOp cosine_op(n);
    
    // 预热
    for (int i = 0; i < 10; i++) {
        cosine_op.compute(a, b);
    }
    
    // GPU性能测试
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        cosine_op.compute(a, b);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // CPU性能测试
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        cpu_cosine_distance(a, b, n);
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "GPU执行 " << iterations << " 次耗时: " << gpu_duration.count() << "ms" << std::endl;
    std::cout << "CPU执行 " << iterations << " 次耗时: " << cpu_duration.count() << "ms" << std::endl;
    std::cout << "GPU加速比: " << (float)cpu_duration.count() / gpu_duration.count() << "x" << std::endl;
    
    delete[] a;
    delete[] b;
    std::cout << "✓ 性能测试完成" << std::endl << std::endl;
}

int main() {
    std::cout << "开始CosineDistanceOp单元测试..." << std::endl << std::endl;
    
    try {
        test_basic_cosine_distance();
        test_same_vectors();
        test_opposite_vectors();
        test_zero_vectors();
        test_large_vectors();
        test_performance();
        
        std::cout << "🎉 所有CosineDistanceOp测试通过！" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
}
