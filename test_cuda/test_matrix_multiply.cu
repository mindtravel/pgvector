#include "../cuda/matrix-multiply.h"
#include "test_utils.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include <cstring>
#include <time.h>
#include <stdlib.h>
#define EPSILON 1e-2

// 测试辅助函数：生成随机矩阵（列主序）
float* generate_random_matrix(int rows, int cols) {
    srand(time(0));
    float* matrix = (float*)malloc(rows * cols * sizeof(float));
    
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            matrix[i + j * rows] = (float)rand() / RAND_MAX * 10.0f - 5.0f; // [-5, 5]
        }
    }
    
    return matrix;
}

// 测试辅助函数：生成单位矩阵（列主序）
float* generate_identity_matrix(int n) {
    float* matrix = (float*)calloc(n * n, sizeof(float));
    
    for (int i = 0; i < n; i++) {
        matrix[i + i * n] = 1.0f;
    }
    
    return matrix;
}

// CPU版本的矩阵乘法计算（用于验证）
// 使用与cuBLAS相同的列主序存储格式
void cpu_matrix_multiply(const float* a, const float* b, float* c, 
                        int M, int N, int K, float alpha, float beta) {
    // 初始化结果矩阵
    if (beta == 0.0f) {
        memset(c, 0, M * N * sizeof(float));
    } else if (beta != 1.0f) {
        for (int i = 0; i < M * N; i++) {
            c[i] *= beta;
        }
    }
    
    // 执行矩阵乘法 C = alpha * A * B + beta * C
    // 使用列主序存储：a[i + j*M] 表示第i行第j列
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i + k * M] * b[k + j * K];
            }
            c[i + j * M] += alpha * sum;
        }
    }
}

void cpu_matrix_multiply_2D(float** a, float** b, float** c, 
    int M, int N, int K, float alpha, float beta) {
    // 初始化结果矩阵
    if (beta == 0.0f) {
        for (int i = 0; i < M; i++) {
            for(int j = 0; j < N; ++j) {
                c[i][j] = 0.0f;
            }
        }
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for(int j = 0; j < N; ++j) {
                c[i][j] *= beta;
            }
        }
    }

    // 执行矩阵乘法 C = alpha * A * B + beta * C
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i][k] * b[j][k];
            }
            c[i][j] += alpha * sum;
        }
    }
}

// 测试1：基本矩阵乘法计算
void test_basic_matrix_multiply(int M, int N, int K) {
    std::cout << "=== Test1: 基本矩阵乘法测试 ===" << std::endl;
    
    float alpha = 1.0f, beta = 0.0f;
    
    std::cout << "测试矩阵大小: " << M << "x" << K << " × " << K << "x" << N << std::endl;
    
    // 计算内存使用量
    size_t memory_mb = (M * K + K * N + M * N) * sizeof(float) / (1024 * 1024);
    std::cout << "内存使用量: " << memory_mb << " MB" << std::endl;
    
    // 生成测试数据
    float** h_A = generate_vector_list(M, K);
    float** h_B = generate_vector_list(N, K);
    float** h_C_gpu = malloc_vector_list(M, N);
    float** h_C_cpu = malloc_vector_list(M, N);
    
    if (DEBUG){
        std::cout << "A" << std::endl;

        for(int i=0; i<M; ++i){
            for(int j=0; j<K; ++j)
                std::cout << h_A[i][j] << " ";
            std::cout << std::endl;        
        }


        std::cout << std::endl;
        std::cout << "B" << std::endl;

        for(int i=0; i<N; ++i){
            for(int j=0; j<K; ++j)
                std::cout << h_B[i][j] << " ";
            std::cout << std::endl;        
        }
    }



    // GPU计算

    auto start = std::chrono::high_resolution_clock::now();
    cuda_sgemmNN_ours(h_A, h_B, h_C_gpu, M, N, K, alpha, beta);
    auto end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // CPU计算
    start = std::chrono::high_resolution_clock::now();
    cpu_matrix_multiply_2D(h_A, h_B, h_C_cpu, M, N, K, alpha, beta);
    end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // 验证结果
    assert(matrix_equal_2D(h_C_gpu, h_C_cpu, M, N, EPSILON));
    
    // 计算性能指标
    float gflops = 2.0f * M * N * K / (gpu_duration.count() / 1000.0f) / 1e9;
    float speedup = (float)cpu_duration.count() / gpu_duration.count();
    
    std::cout << "GPU耗时: " << gpu_duration.count() << " ms" << std::endl;
    std::cout << "CPU耗时: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "加速比: " << speedup << "x" << std::endl;
    std::cout << "GPU性能: " << gflops << " GFLOPS" << std::endl;
    
    // 清理内存
    free_vector_list(h_A);
    free_vector_list(h_B);
    free_vector_list(h_C_gpu);
    free_vector_list(h_C_cpu);
}

// 测试2：单位矩阵测试
void test_identity_matrix() {
    std::cout << "=== Test2: 单位矩阵测试 ===" << std::endl;
    
    int M = 64, N = 64, K = 64;
    float alpha = 1.0f, beta = 0.0f;
    
    // 生成单位矩阵
    float* h_A = generate_identity_matrix(K);
    float* h_B = generate_random_matrix(K, N);
    float* h_C_gpu = (float*)calloc(M * N, sizeof(float));
    float* h_C_cpu = (float*)calloc(M * N, sizeof(float));
    
    // GPU计算
    cuda_sgemmNN(h_A, h_B, h_C_gpu, M, N, K, alpha, beta);
    
    // CPU计算
    cpu_matrix_multiply(h_A, h_B, h_C_cpu, M, N, K, alpha, beta);
    
    // 验证结果（A是单位矩阵，所以结果应该等于B）
    assert(matrix_equal(h_C_gpu, h_B, M, N, EPSILON));
    
    std::cout << "单位矩阵测试通过 ✓" << std::endl << std::endl;
    
    // 清理内存
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
}

// 测试3：不同alpha和beta值测试
void test_alpha_beta_values() {
    std::cout << "=== Test3: Alpha/Beta值测试 ===" << std::endl;
    
    int M = 32, N = 32, K = 32;
    float alpha = 2.5f, beta = 1.5f;
    
    float* h_A = generate_random_matrix(M, K);
    float* h_B = generate_random_matrix(K, N);
    float* h_C_initial = generate_random_matrix(M, N);
    
    float* h_C_gpu = (float*)malloc(M * N * sizeof(float));
    float* h_C_cpu = (float*)malloc(M * N * sizeof(float));
    
    // 复制初始值
    memcpy(h_C_gpu, h_C_initial, M * N * sizeof(float));
    memcpy(h_C_cpu, h_C_initial, M * N * sizeof(float));
    
    // GPU计算
    cuda_sgemmNN(h_A, h_B, h_C_gpu, M, N, K, alpha, beta);
    
    // CPU计算
    cpu_matrix_multiply(h_A, h_B, h_C_cpu, M, N, K, alpha, beta);
    
    // 验证结果
    assert(matrix_equal(h_C_gpu, h_C_cpu, M, N, EPSILON));
    
    std::cout << "Alpha=" << alpha << ", Beta=" << beta << " 测试通过 ✓" << std::endl << std::endl;
    
    // 清理内存
    free(h_A);
    free(h_B);
    free(h_C_initial);
    free(h_C_gpu);
    free(h_C_cpu);
}

// 测试4：压力测试
void test_large_scale_stress(int M, int N, int K) {
    std::cout << "=== Test4: 大规模压力测试 ===" << std::endl;
    
    
    float alpha = 1.0f, beta = 0.0f;
    
    std::cout << "测试矩阵大小: " << M << "x" << K << " × " << K << "x" << N << std::endl;
    
    // 计算内存使用量
    size_t memory_mb = (M * K + K * N + M * N) * sizeof(float) / (1024 * 1024);
    std::cout << "内存使用量: " << memory_mb << " MB" << std::endl;
    
    // 生成测试数据
    float** h_A = generate_vector_list(M, K);
    float** h_B = generate_vector_list(N, K);
    float** h_C_gpu = malloc_vector_list(M, N);
    float** h_C_cpu = malloc_vector_list(M, N);


    // GPU计算
    auto start = std::chrono::high_resolution_clock::now();
    cuda_sgemmNN_ours(h_A, h_B, h_C_gpu, M, N, K, alpha, beta);
    auto end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // CPU计算
    start = std::chrono::high_resolution_clock::now();
    cpu_matrix_multiply_2D(h_A, h_B, h_C_cpu, M, N, K, alpha, beta);
    end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // 验证结果
    assert(matrix_equal_2D(h_C_gpu, h_C_cpu, M, N, EPSILON));
    
    // 计算性能指标
    float gflops = 2.0f * M * N * K / (gpu_duration.count() / 1000.0f) / 1e9;
    float speedup = (float)cpu_duration.count() / gpu_duration.count();
    
    std::cout << "GPU耗时: " << gpu_duration.count() << " ms" << std::endl;
    std::cout << "CPU耗时: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "加速比: " << speedup << "x" << std::endl;
    std::cout << "GPU性能: " << gflops << " GFLOPS" << std::endl;
    
    // 清理内存
    free_vector_list(h_A);
    free_vector_list(h_B);
    free_vector_list(h_C_gpu);
    free_vector_list(h_C_cpu);
    
    std::cout << "大规模压力测试完成 ✓" << std::endl << std::endl;
}

int main() {
    srand(time(0));
    std::cout << "开始矩阵乘法单元测试..." << std::endl << std::endl;
    
    try {
        // test_basic_matrix_multiply(10, 100, 30);
        test_basic_matrix_multiply(1024, 1024, 1024);
        // test_identity_matrix();
        // test_alpha_beta_values();
        // test_large_scale_stress(1024, 1024, 1024);
        
        std::cout << "all_test_passed" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
}
