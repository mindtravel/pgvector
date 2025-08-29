#include <cuda_runtime.h>
#include "martix-multiply.h"

// 声明核函数
__global__ void sgemmNN(
    const float *A, int lda,
    const float *B, int ldb,
    float* C, int ldc,
    int k, float alpha, float beta)
{
    // 计算数据指针偏移
    A += blockIdx.x * 64 + threadIdx.x + threadIdx.y*16;
    B += threadIdx.x + ( blockIdx.y * 16 + threadIdx.y ) * ldb;
    C += blockIdx.x * 64 + threadIdx.x + (threadIdx.y + blockIdx.y * ldc ) * 16;
    
    // 声明片上存储
    __shared__ float bs[16][17];
    float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
    const float *Blast = B + k;
    
    do {
        #pragma unroll
        for( int i = 0; i < 16; i += 4 )
            bs[threadIdx.x][threadIdx.y+i] = B[i*ldb];
        
        B += 16;
        __syncthreads();
        
        // 瓶颈：读取A的列
        #pragma unroll
        for( int i = 0; i < 16; i++, A += lda ) {
            // 进行秩-1更新
            c[0] += A[0]*bs[i][0]; c[1] += A[0]*bs[i][1]; c[2] += A[0]*bs[i][2]; c[3] += A[0]*bs[i][3];
            c[4] += A[0]*bs[i][4]; c[5] += A[0]*bs[i][5]; c[6] += A[0]*bs[i][6]; c[7] += A[0]*bs[i][7];
            c[8] += A[0]*bs[i][8]; c[9] += A[0]*bs[i][9]; c[10] += A[0]*bs[i][10];c[11] += A[0]*bs[i][11];
            c[12] += A[0]*bs[i][12];c[13] += A[0]*bs[i][13];c[14] += A[0]*bs[i][14];c[15] += A[0]*bs[i][15];
        }
        __syncthreads();
    } while( B < Blast );
    
    // 将C的块存储到内存
    for( int i = 0; i < 16; i++, C += ldc )
        C[0] = alpha*c[i] + beta*C[0];
}

// CUDA接口实现
void cuda_sgemmNN(const float* h_A, const float* h_B, float* h_C,
                  int M, int N, int K, float alpha, float beta) {
    int lda = K, ldb = N, ldc = N;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // 假设 M 是 64 的倍数，N 是 16 的倍数
    dim3 blockDim(16, 16);
    dim3 gridDim(M / 64, N / 16);

    sgemmNN<<<gridDim, blockDim>>>(d_A, lda, d_B, ldb, d_C, ldc, K, alpha, beta);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}