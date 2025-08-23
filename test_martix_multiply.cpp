#include <iostream>
#include <cstdlib>
#include "martix-multiply.h"

#define M 64
#define N 16
#define K 64

int main() {
    size_t sizeA = M * K;
    size_t sizeB = K * N;
    size_t sizeC = M * N;

    float* h_A = new float[sizeA];
    float* h_B = new float[sizeB];
    float* h_C = new float[sizeC];

    // 初始化输入矩阵
    for (size_t i = 0; i < sizeA; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < sizeB; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // 调用CUDA矩阵乘法算子
    cuda_sgemmNN(h_A, h_B, h_C, M, N, K, 1.0f, 0.0f);

    // 简单验证
    std::cout << "C[0] = " << h_C[0] << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    return 0;
}