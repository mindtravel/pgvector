#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "matrix-multiply.h"
#include "kernels.h"
#include "cudatimer.h"
#define ENABLE_CUDA_TIMING 0 /*是否启用CUDATimer计时*/

/**
 * 矩阵乘法接口
 */
void cuda_sgemmNN(float* h_A, float* h_B, float* h_C,
                  int M, int N, int K, float alpha, float beta) {

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // cuBLAS句柄
    cublasHandle_t handle;

    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    {
        CUDATimer timer_manage("GPU Memory Allocation", ENABLE_CUDA_TIMING, false);
        cublasCreate(&handle);
        cudaMalloc(&d_A, sizeA);
        cudaMalloc(&d_B, sizeB);
        cudaMalloc(&d_C, sizeC);
    }

    // 复制数据到设备
    {
        CUDATimer timer_trans1("H2D Data Transfer", ENABLE_CUDA_TIMING);
        cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);        
    }


    /* 核函数执行 */
    {
        CUDATimer timer_compute("Kernel Execution", ENABLE_CUDA_TIMING);
        /**
         * 使用cuBLAS进行矩阵乘法
         * cuBLAS默认使用列主序，leading dimension是行数
         * */ 
        cublasSgemm(handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置 A 和 B
                    M, N, K,                   // C 的维度 M×N
                    &alpha, 
                    d_A, M,                    // lda = M（A 的leading dimension）
                    d_B, K,                    // ldb = K（B 的leading dimension）
                    &beta, 
                    d_C, M);                   // ldc = M（C 的leading dimension）

        // 同步并复制结果回主机
        cudaDeviceSynchronize(); 
    }


    {
        CUDATimer timer_trans2("D2H Data Transfer", ENABLE_CUDA_TIMING);
        cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    }

    {
        CUDATimer timer_manage2("GPU Memory Free", ENABLE_CUDA_TIMING, false);
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);    
    }
}

/**
 * 我们的矩阵乘法接口 [n_querys, n_dim] * [n_batch * n_dim].T
 */
void cuda_sgemmNN_ours(float** h_A, float** h_B, float** h_C,
    int M, int N, int K, float alpha, float beta) {

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // cuBLAS句柄
    cublasHandle_t handle;

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    {
        CUDATimer timer_manage("GPU Memory Allocation", ENABLE_CUDA_TIMING, false);
        cublasCreate(&handle);
        cudaMalloc(&d_A, sizeA);
        cudaMalloc(&d_B, sizeB);
        cudaMalloc(&d_C, sizeC);
    }

    // 复制数据到设备
    {
        CUDATimer timer_trans1("H2D Data Transfer", ENABLE_CUDA_TIMING);
        // 使用简单的cudaMemcpy，因为数据已经是连续存储的
        cudaMemcpy(d_A, h_A[0], sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B[0], sizeB, cudaMemcpyHostToDevice);
    }


    /* 核函数执行 */
    {
        CUDATimer timer_compute("Kernel Execution", ENABLE_CUDA_TIMING);
        /**
         * 使用cuBLAS进行矩阵乘法
         * 计算: [M, K] * [N, K].T = [M, N]
         * 即: A * B.T = C
         * 
         * 参数说明:
         * - A: [M, K] 矩阵，查询向量组，行主序存储
         * - B: [N, K] 矩阵，数据向量组，行主序存储
         * - C: [M, N] 矩阵，结果矩阵，行主序存储
         * - cuBLAS使用列主序，所以leading dimension是行数
         */
        cublasSgemm(handle, 
                    CUBLAS_OP_T, CUBLAS_OP_N, 
                    N, M, K,                   
                    &alpha, 
                    d_B, K,            
                    d_A, K,               
                    &beta, 
                    d_C, N);                

        // 同步并复制结果回主机
        cudaDeviceSynchronize(); 
    }


    {
        CUDATimer timer_trans2("D2H Data Transfer", ENABLE_CUDA_TIMING);
        // 使用简单的cudaMemcpy，因为数据是连续存储的
        cudaMemcpy(h_C[0], d_C, sizeC, cudaMemcpyDeviceToHost);
    }

    {
        CUDATimer timer_manage2("GPU Memory Free", ENABLE_CUDA_TIMING, false);
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);    
    }

}