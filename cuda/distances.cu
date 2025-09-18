#include "distances.h"
#include <math.h>
#include <assert.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include "cudatimer.h"
#include "kernels.h"
#define ENABLE_CUDA_TIMING 1 /*是否启用CUDATimer计时*/

/**
 * 计算一个batch的query和待检索的list之间的余弦距离 [n_querys, n_dim] * [n_batch * n_dim].T
 */
void cuda_cosine_dist(float** query_vector_group_cpu, float** data_vector_group_cpu, float** h_cos_dist,
    int n_query, int n_batch, int n_dim, float alpha, float beta) {

    const int NUM_STREAMS = 0; // cuda流的数量
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

        l2_norm_kernel<<<query_gridDim, vector_dim, n_dim * sizeof(float)>>>(
            d_query_vectors, d_query_norm, 
            n_query, n_dim
        );

        l2_norm_kernel<<<data_gridDim, vector_dim, n_dim * sizeof(float)>>>(
            d_data_vectors, d_data_norm, 
            n_batch, n_dim
        );        
        
        // 调试信息：打印GPU内存指针地址（不要访问内容）
        // std::cout << "GPU memory pointers: " << d_query_norm << " " << d_data_norm << " " << d_cos_dist << std::endl;

        /**
        * 使用cuBLAS进行矩阵乘法
        * cuBLAS默认使用列主序，leading dimension是行数
        * */ 
       cublasSgemm(handle, 
            CUBLAS_OP_T, CUBLAS_OP_N, 
            n_batch, n_query, n_dim,                   
            &alpha, 
            d_data_vectors, n_dim,            
            d_query_vectors, n_dim,               
            &beta, 
            d_cos_dist, n_batch
        );    


        cudaDeviceSynchronize(); 

        cos_distance_kernel<<<data_gridDim, query_gridDim>>>(
            d_query_norm, d_data_norm, d_cos_dist,
            n_query, n_batch, n_dim
        );
        // cos_sim_kernal<<<data_gridDim, query_gridDim>>>(
        //     d_query_norm, d_data_norm, d_cos_dist,
        //     n_query, n_batch, n_dim
        // );
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


void cuda_cosine_dist_topk(float** query_vector_group_cpu, float** data_vector_group_cpu, 
    int* data_index, int** topk_index,
    int n_query, int n_batch, int n_dim,
    int k /*查找的最近邻个数*/
){ 
    /**
    * 对一个batch的查询向量，找出余弦距离最近的topk，返回一个形状为 [batch, k] 的索引矩阵
    **/

    float alpha = 1.0f; 
    float beta = 0.0f;

    const int NUM_STREAMS = 0; // cuda流的数量
    bool query_copied = false; // query常驻显存

    dim3 query_gridDim(n_query);
    dim3 data_gridDim(n_batch);
    dim3 vector_dim(n_dim);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
    
    size_t size_query = n_query * n_dim * sizeof(float);
    size_t size_idx = n_batch * sizeof(int);
    size_t size_data = n_batch * n_dim * sizeof(float);
    size_t size_cos_dist = n_query * n_batch * sizeof(float);
    size_t size_topk_idx = n_batch * (k + n_batch) * sizeof(int);
    size_t size_topk_dist = n_batch * (k + n_batch) * sizeof(float);

    // cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);
    // cublasSetStream(handle, streams[0]); 

    // 分配设备内存
    float *d_query_vectors, *d_data_vectors, *d_cos_dist,
        *d_query_norm, *d_data_norm, *d_topk_dist;
    int *d_index, *d_topk_index;
    {
        CUDATimer timer_manage("GPU Memory Allocation", ENABLE_CUDA_TIMING, false);
        cudaMalloc(&d_query_vectors, size_query);
        cudaMalloc(&d_index, size_idx);
        cudaMalloc(&d_data_vectors, size_data);
        cudaMalloc(&d_topk_index, size_topk_idx);/*存储各个query的top-k向量索引*/
        cudaMalloc(&d_topk_dist, size_topk_dist);/*存储各个query的top-k向量索引*/
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

        cudaMemcpy(d_query_vectors, query_vector_group_cpu, size_query, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_data_vectors, data_vector_group_cpu, size_data, cudaMemcpyHostToDevice);        
    }


    /* 核函数执行 */
    {
        CUDATimer timer_compute("Kernel Execution", ENABLE_CUDA_TIMING);

        l2_norm_kernel<<<query_gridDim, vector_dim, n_dim * sizeof(float)>>>(
            d_query_vectors, d_query_norm, 
            n_query, n_dim
        );

        l2_norm_kernel<<<data_gridDim, vector_dim, n_dim * sizeof(float)>>>(
            d_data_vectors, d_data_norm, 
            n_batch, n_dim
        );        
        
        // 调试信息：打印GPU内存指针地址（不要访问内容）
        // std::cout << "GPU memory pointers: " << d_query_norm << " " << d_data_norm << " " << d_cos_dist << std::endl;

        /**
        * 使用cuBLAS进行矩阵乘法
        * cuBLAS默认使用列主序，leading dimension是行数
        * */ 
       cublasSgemm(handle, 
            CUBLAS_OP_T, CUBLAS_OP_N, 
            n_batch, n_query, n_dim,                   
            &alpha, 
            d_data_vectors, n_dim,            
            d_query_vectors, n_dim,               
            &beta, 
            d_cos_dist, n_batch
        );    


        cudaDeviceSynchronize(); 

        cos_distance_kernel<<<data_gridDim, query_gridDim>>>(
            d_query_norm, d_data_norm, d_cos_dist,
            n_query, n_batch, n_dim
        );
        
        cudaDeviceSynchronize(); 

        topk_kernel<<<data_gridDim, query_gridDim>>>(
            d_index, d_cos_dist, 
            d_topk_index, d_topk_dist,
            n_query, n_batch, k
        );
    }


    {
        CUDATimer timer_trans2("D2H Data Transfer", ENABLE_CUDA_TIMING);
        cudaMemcpy2D(
            topk_index[0],
            k * sizeof(int), /*每行只复制topk结果*/
            d_topk_index,
            (k + n_batch) * sizeof(int), /*每行的间隔是 (k+n_batch)单位 */
            (k + n_batch) * sizeof(int),
            n_query,
            cudaMemcpyDeviceToHost
        );
        // cudaMemcpy(topk_index, d_topk_index, size_topk_idx, cudaMemcpyDeviceToHost);        

    }

    {
        CUDATimer timer_manage2("GPU Memory Free", ENABLE_CUDA_TIMING, false);
        cublasDestroy(handle);
        cudaFree(d_query_vectors);
        cudaFree(d_data_vectors);
        cudaFree(d_cos_dist);
        cudaFree(d_query_norm);
        cudaFree(d_data_norm);
        cudaFree(d_topk_index);
        
        // 销毁CUDA流
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }
}