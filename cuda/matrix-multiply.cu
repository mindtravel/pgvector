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

/**
 * 计算一个batch的query和待检索的list之间的余弦距离 [n_querys, n_dim] * [n_batch * n_dim].T
 */
void cuda_cosine_dist(float** query_vector_group_cpu, float** data_vector_group_cpu, float** h_cos_dist,
    int n_query, int n_batch, int n_dim, float alpha, float beta) {

    const int NUM_STREAMS = 2; // cuda流的数量
    bool query_copied = false; // query常驻显存

    dim3 query_gridDim(n_query);
    dim3 data_gridDim(n_batch);
    dim3 vector_dim(n_dim);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
    
    size_t size_query = n_query * n_dim * sizeof(float);
    size_t size_data = n_batch * n_dim * sizeof(float);
    size_t size_cos_dist = n_query * n_batch * sizeof(float);

    // cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);
    // cublasSetStream(handle, streams[0]); 

    // 分配设备内存
    float *d_query_vectors, *d_data_vectors, *d_cos_dist,
        *d_query_norm, *d_data_norm;
    {
        CUDATimer timer_manage("GPU Memory Allocation", ENABLE_CUDA_TIMING, false);
        cudaMalloc(&d_query_vectors, size_query);
        cudaMalloc(&d_data_vectors, size_data);
        cudaMalloc(&d_cos_dist, size_cos_dist);/*存储cos距离矩阵*/
        // cudaMalloc((void**)&d_dot_sum_, gridSize_ * sizeof(float));
        cudaMalloc(&d_query_norm, n_query * sizeof(float)); /*存储query的l2 Norm*/
        cudaMalloc(&d_data_norm, n_batch * sizeof(float)); /*存储data的l2 Norm*/
        // cudaMalloc((void**)&d_sq_b_sum_, gridSize_ * sizeof(float));
    }

    // 复制数据到设备
    {
        CUDATimer timer_trans1("H2D Data Transfer", ENABLE_CUDA_TIMING);
        if(query_copied == false){
            cudaMemcpy2D(
                d_query_vectors,
                n_dim * sizeof(float),
                query_vector_group_cpu[0],
                n_dim * sizeof(float),
                n_dim * sizeof(float),
                n_query,
                cudaMemcpyHostToDevice
            );            
            query_copied = true;
        }

        cudaMemcpy2D(
            d_data_vectors,
            n_dim * sizeof(float),
            data_vector_group_cpu[0],
            n_dim * sizeof(float),
            n_dim * sizeof(float),
            n_batch,
            cudaMemcpyHostToDevice
        );
        // cudaMemcpy(d_query_vectors, query_vector_group_cpu, size_query, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_data_vectors, data_vector_group_cpu, size_data, cudaMemcpyHostToDevice);        
    }


    /* 核函数执行 */
    {
        CUDATimer timer_compute("Kernel Execution", ENABLE_CUDA_TIMING);
        /**
        * 使用cuBLAS进行矩阵乘法
        * cuBLAS默认使用列主序，leading dimension是行数
        * */ 
        cublasSgemm(handle, 
            CUBLAS_OP_T, CUBLAS_OP_N,  // 不转置 A，转置 B
            n_query, n_batch, n_dim,   // C 的维度 M×N
            &alpha, 
            d_query_vectors, n_query,  // lda = n_query（A 的leading dimension）
            d_data_vectors, n_batch,   // ldb = n_batch（B转置后的leading dimension）
            &beta, 
            d_cos_dist, n_query);             // ldc = n_query（C 的leading dimension）


        l2_norm_kernel<<<query_gridDim, vector_dim, n_dim * sizeof(float)>>>(
            d_query_vectors, d_query_norm, 
            n_query, n_dim
        );

        l2_norm_kernel<<<data_gridDim, vector_dim, n_dim * sizeof(float)>>>(
            d_data_vectors, d_data_norm, 
            n_batch, n_dim
        );        
        
        for(int i=0; i< n_query; ++i)
            std::cout << d_query_norm[i] << " ";
        std::cout << std::endl;

        for(int i=0; i< n_batch; ++i)
            std::cout << d_data_norm[i] << " ";
        std::cout << std::endl;

        cudaDeviceSynchronize(); 

        cos_distance_kernel<<<data_gridDim, query_gridDim>>>(
            d_query_norm, d_data_norm, d_cos_dist,
            n_query, n_batch, n_dim
        );

        cudaDeviceSynchronize(); 
    }


    {
        CUDATimer timer_trans2("D2H Data Transfer", ENABLE_CUDA_TIMING);
        cudaMemcpy2D(
            h_cos_dist[0],
            n_batch * sizeof(float),
            d_cos_dist,
            n_batch * sizeof(float),
            n_batch * sizeof(float),
            n_query,
            cudaMemcpyDeviceToHost
        );

    }

    {
        CUDATimer timer_manage2("GPU Memory Free", ENABLE_CUDA_TIMING, false);
        cublasDestroy(handle);
        cudaFree(d_query_vectors);
        cudaFree(d_data_vectors);
        cudaFree(d_cos_dist);
        cudaFree(d_query_norm);
        cudaFree(d_data_norm);
        
        // 销毁CUDA流
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }

}