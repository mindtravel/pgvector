#include "distances.h"
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#include "kernels.h"
#include "select_topk.cuh"

#include "../test_cuda/test_utils.cuh"
#include "pch.h"


#define ENABLE_CUDA_TIMING 1 /*是否启用CUDATimer计时*/

/**
 * 计算一个batch的query和待检索的list之间的余弦距离 [n_querys, n_dim] * [n_batch * n_dim].T
 */
void cuda_cosine_dist(float** h_query_vectors, float** h_data_vectors, float** h_cos_dist,
    int n_query, int n_batch, int n_dim, float alpha, float beta) {

    const int NUM_STREAMS = 0; // cuda流的数量
    bool query_copied = false; // query常驻显存

    dim3 queryDim(n_query);
    dim3 dataDim(n_batch);
    dim3 vectorDim(n_dim);

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
    cudaSetDevice(0);
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
                h_query_vectors[0],
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
            h_data_vectors[0],
            n_dim * sizeof(float),
            n_dim * sizeof(float),
            n_batch,
            cudaMemcpyHostToDevice
        );
        // cudaMemcpy(d_query_vectors, h_query_vectors, size_query, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_data_vectors, h_data_vectors, size_data, cudaMemcpyHostToDevice);        
    }


    /* 核函数执行 */
    {
        CUDATimer timer_compute("Kernel Execution", ENABLE_CUDA_TIMING);

        l2_norm_kernel<<<queryDim, vectorDim, n_dim * sizeof(float)>>>(
            d_query_vectors, d_query_norm, 
            n_query, n_dim
        );

        l2_norm_kernel<<<dataDim, vectorDim, n_dim * sizeof(float)>>>(
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

        cos_distance_kernel<<<dataDim, queryDim>>>(
            d_query_norm, d_data_norm, d_cos_dist,
            n_query, n_batch, n_dim
        );
        // cos_sim_kernal<<<dataDim, queryDim>>>(
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


void cuda_cosine_dist_topk(
    float** h_query_vectors, float** h_data_vectors, 
    int* data_index, int** topk_index,
    int n_query, int n_batch, int n_dim,
    int k /*查找的最近邻个数*/
){ 
    /**
    * 对一个batch的查询向量，找出余弦距离最近的topk，返回一个形状为 [batch, k] 的索引矩阵
    **/
//    table_2D("topk_index", topk_index, n_query, k);

    float alpha = 1.0f; 
    float beta = 0.0f;

    const int NUM_STREAMS = 0; // cuda流的数量
    bool query_copied = false; // query常驻显存

    dim3 queryDim(n_query);
    dim3 dataDim(n_batch);
    dim3 vectorDim(n_dim);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
    
    size_t size_query = n_query * n_dim * sizeof(float);
    // size_t size_idx = n_batch * sizeof(int);
    size_t size_data = n_batch * n_dim * sizeof(float);
    size_t size_dist = n_query * (n_batch + k) * sizeof(float);
    size_t size_idx = n_query * (n_batch + k) * sizeof(int);
    // size_t size_topk_dist = n_query * k * sizeof(float);

    // cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);
    // cublasSetStream(handle, streams[0]); 

    // 分配设备内存
    float *d_query_vectors, *d_data_vectors, *d_cos_dist, *d_topk_cos_dist,
        *d_query_norm, *d_data_norm;
    int *d_index, *d_topk_index;
    {
        CUDATimer timer_manage("GPU Memory Allocation", ENABLE_CUDA_TIMING, false);
        cudaMalloc(&d_query_vectors, size_query);
        cudaMalloc(&d_data_vectors, size_data);
        cudaMalloc(&d_cos_dist, size_dist);/*存储各个query需要查找的data向量的距离*/
        cudaMalloc(&d_index, size_idx);/*存储各个query需要查找的data向量的索引*/
        cudaMalloc(&d_topk_cos_dist, size_topk_dist);/*存储topk距离*/
        cudaMalloc(&d_topk_index, size_topk_idx);/*存储topk索引*/

        cudaMalloc(&d_query_norm, n_query * sizeof(float)); /*存储query的l2 Norm*/
        cudaMalloc(&d_data_norm, n_batch * sizeof(float)); /*存储data的l2 Norm*/
    }

    // 复制数据到设备
    {

        CUDATimer timer_trans1("H2D Data Transfer", ENABLE_CUDA_TIMING);
        // 复制查询向量，然后常驻
        if(query_copied == false){
            cudaMemcpy2D(
                d_query_vectors,
                n_dim * sizeof(float),
                h_query_vectors[0],
                n_dim * sizeof(float),
                n_dim * sizeof(float),
                n_query,
                cudaMemcpyHostToDevice
            );            
            query_copied = true;
        }

        /* 复制data向量 */
        cudaMemcpy2D(
            d_data_vectors,
            n_dim * sizeof(float),
            h_data_vectors[0],
            n_dim * sizeof(float),
            n_dim * sizeof(float),
            n_batch,
            cudaMemcpyHostToDevice
        );
        // cudaMemcpy(d_data_vectors, h_data_vectors, size_data, cudaMemcpyHostToDevice);        

        /* 复制索引数组 */
        cudaMemcpy2D(
            d_index,
            n_batch * sizeof(float),
            data_index,
            0,/* spitch设为0，意为每次都复制同一行 */
            n_batch * sizeof(float),
            n_query,
            cudaMemcpyHostToDevice
        );
        /* 初始化距离数组（为一个小于-1的负数） */
        cudaMemset((void*)d_topk_cos_dist, (int)0xFE, n_query * k * sizeof(float));
    }

    // print_cuda_2D("index matrix", d_index, n_query, n_batch);
    // print_cuda_2D("cos distance matrix", d_cos_dist, n_query, n_batch);
    // print_2D("query vector on host", h_query_vectors, n_query, n_dim);
    // print_2D("data vector on host", h_data_vectors, n_batch, n_dim);
    // print_cuda_2D("query vector on device", d_query_vectors, n_query, n_dim);
    // print_cuda_2D("data vector on device", d_data_vectors, n_batch, n_dim);
    // print_cuda_2D("topk index matrix", d_topk_index, n_query, k);
    // print_cuda_2D("topk cos distance matrix", d_topk_cos_dist, n_query, k);

    /* 核函数执行 */
    {
        CUDATimer timer_compute("Kernel Execution", ENABLE_CUDA_TIMING);

        l2_norm_kernel<<<queryDim, vectorDim, n_dim * sizeof(float)>>>(
            d_query_vectors, d_query_norm, 
            n_query, n_dim
        );

        l2_norm_kernel<<<dataDim, vectorDim, n_dim * sizeof(float)>>>(
            d_data_vectors, d_data_norm, 
            n_batch, n_dim
        );        

       
        table_cuda_1D("data_norm", d_data_norm, n_batch);
        // table_cuda_2D("data vectors", d_data_vectors, n_batch, n_dim);

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

        cos_distance_kernel<<<dataDim, queryDim>>>(
            d_query_norm, d_data_norm, d_cos_dist,
            n_query, n_batch, n_dim
        );
        
        // table_cuda_2D("cos distance matrix", d_cos_dist, n_query, n_batch);

        cudaDeviceSynchronize(); 
        
        // 设置CUDA设备以确保RAFT库使用正确的设备
        
        raft::resources handle;

        // 创建输入矩阵视图
        auto d_cos_dist_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
            d_cos_dist, n_query, n_batch);
        auto d_index_view = raft::make_device_matrix_view<const int, int64_t, raft::row_major>(
            d_index, n_query, n_batch);
        
        // 创建输出矩阵视图
        auto d_topk_cos_dist_view = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
            d_topk_cos_dist, n_query, k);
        auto d_topk_index_view = raft::make_device_matrix_view<int, int64_t, raft::row_major>(
            d_topk_index, n_query, k);

        COUT_ENDL("here");

        select_k(handle,
            d_cos_dist_view,
            std::make_optional(d_index_view),
            d_topk_cos_dist_view,
            d_topk_index_view,
            true,  // select_min = true
            false, // sorted = false
            SelectAlgo::kAuto,
            std::nullopt);

        table_2D("topk_index", topk_index, n_query, k);
    }


    {
        CUDATimer timer_trans2("D2H Data Transfer", ENABLE_CUDA_TIMING);
        cudaMemcpy(topk_index, d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost);        
    }

    {
        CUDATimer timer_manage2("GPU Memory Free", ENABLE_CUDA_TIMING, false);
        cublasDestroy(handle);
        cudaFree(d_query_vectors);
        cudaFree(d_data_vectors);
        cudaFree(d_cos_dist);
        cudaFree(d_query_norm);
        cudaFree(d_data_norm);
        cudaFree(d_index);
        
        // 销毁CUDA流
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }
}