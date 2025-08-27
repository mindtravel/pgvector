#include "../cuda/distances.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstring>
#include <chrono>

// 测试辅助函数：比较浮点数
bool float_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// 测试辅助函数：比较向量
bool vector_equal(const float* a, const float* b, int n, float epsilon = 1e-5f) {
    for (int i = 0; i < n; i++) {
        if (!float_equal(a[i], b[i], epsilon)) {
            return false;
        }
    }
    return true;
}

// 测试1：基本归一化功能
void test_basic_normalization() {
    std::cout << "=== 测试1：基本归一化功能 ===" << std::endl;
    
    VectorNormalizer normalizer;
    float data[] = {3.0f, 4.0f, 0.0f, 5.0f};
    int n = 4;
    
    std::cout << "原始向量: ";
    for (int i = 0; i < n; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    
    // 计算期望的模长
    float expected_norm = sqrtf(3.0f*3.0f + 4.0f*4.0f + 0.0f*0.0f + 5.0f*5.0f);
    std::cout << "期望模长: " << expected_norm << std::endl;
    
    normalizer.normalize(data, n);
    
    std::cout << "归一化后向量: ";
    for (int i = 0; i < n; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    
    // 验证模长
    float actual_norm = normalizer.last_norm();
    std::cout << "实际模长: " << actual_norm << std::endl;
    
    assert(float_equal(actual_norm, expected_norm));
    
    // 验证归一化后的向量模长为1
    float normalized_norm = sqrtf(data[0]*data[0] + data[1]*data[1] + data[2]*data[2] + data[3]*data[3]);
    std::cout << "归一化后向量模长: " << normalized_norm << std::endl;
    
    assert(float_equal(normalized_norm, 1.0f));
    
    std::cout << "✓ 基本归一化测试通过" << std::endl << std::endl;
}

// 测试2：零向量处理
void test_zero_vector() {
    std::cout << "=== 测试2：零向量处理 ===" << std::endl;
    
    VectorNormalizer normalizer;
    float data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    int n = 4;
    
    std::cout << "零向量: ";
    for (int i = 0; i < n; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    
    normalizer.normalize(data, n);
    
    std::cout << "归一化后: ";
    for (int i = 0; i < n; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    
    // 零向量归一化后应该保持为零向量
    float expected[] = {0.0f, 0.0f, 0.0f, 0.0f};
    assert(vector_equal(data, expected, n));
    
    std::cout << "✓ 零向量测试通过" << std::endl << std::endl;
}

// 测试3：单位向量
void test_unit_vector() {
    std::cout << "=== 测试3：单位向量 ===" << std::endl;
    
    VectorNormalizer normalizer;
    float data[] = {1.0f, 0.0f, 0.0f, 0.0f};
    int n = 4;
    
    std::cout << "单位向量: ";
    for (int i = 0; i < n; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    
    normalizer.normalize(data, n);
    
    std::cout << "归一化后: ";
    for (int i = 0; i < n; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    
    // 单位向量归一化后应该保持不变
    float expected[] = {1.0f, 0.0f, 0.0f, 0.0f};
    assert(vector_equal(data, expected, n));
    
    std::cout << "✓ 单位向量测试通过" << std::endl << std::endl;
}

// 测试4：大维度向量
void test_large_vector() {
    std::cout << "=== 测试4：大维度向量 ===" << std::endl;
    
    const int n = 1024;
    float* data = new float[n];
    
    // 初始化向量
    for (int i = 0; i < n; i++) {
        data[i] = (float)(i + 1);
    }
    
    VectorNormalizer normalizer;
    normalizer.normalize(data, n);
    
    // 验证归一化后的模长
    float norm = 0.0f;
    for (int i = 0; i < n; i++) {
        norm += data[i] * data[i];
    }
    norm = sqrtf(norm);
    
    std::cout << "大维度向量归一化后模长: " << norm << std::endl;
    assert(float_equal(norm, 1.0f));
    
    delete[] data;
    std::cout << "✓ 大维度向量测试通过" << std::endl << std::endl;
}

// 测试5：性能测试
void test_performance() {
    std::cout << "=== 测试5：性能测试 ===" << std::endl;
    
    const int n = 1024;
    const int iterations = 1000;
    float* data = new float[n];
    
    // 初始化向量
    for (int i = 0; i < n; i++) {
        data[i] = (float)(i + 1);
    }
    
    VectorNormalizer normalizer;
    
    // 预热
    float warmup_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 10; i++) {
        memcpy(data, warmup_data, 4 * sizeof(float));
        normalizer.normalize(data, 4);
    }
    
    // 性能测试
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        // 重新初始化数据
        for (int j = 0; j < n; j++) {
            data[j] = (float)(j + 1);
        }
        normalizer.normalize(data, n);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "执行 " << iterations << " 次归一化耗时: " << duration.count() << "ms" << std::endl;
    std::cout << "平均每次归一化耗时: " << (float)duration.count() / iterations << "ms" << std::endl;
    
    delete[] data;
    std::cout << "✓ 性能测试完成" << std::endl << std::endl;
}

int main() {
    std::cout << "开始VectorNormalizer单元测试..." << std::endl << std::endl;
    
    try {
        test_basic_normalization();
        test_zero_vector();
        test_unit_vector();
        test_large_vector();
        test_performance();
        
        std::cout << "🎉 所有VectorNormalizer测试通过！" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
}
