#pragma once


// CUDA矩阵乘法算子接口
void cuda_sgemmNN(const float* h_A, const float* h_B, float* h_C,
                  int M, int N, int K, float alpha, float beta);

                  