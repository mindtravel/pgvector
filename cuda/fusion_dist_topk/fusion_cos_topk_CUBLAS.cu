#include "../utils.cuh"
#include <cfloat>

#include "../l2norm/l2norm.cuh"
#include "fusion_dist_topk_warpsort.cuh"
#include "pch.h"

// 定义最大k值
#define MAX_K 1024
#define ENABLE_CUDA_TIMING 1 /*是否启用CUDATimer计时*/

void cuda_cos_dist_topk(
    float** h_query_vectors, float** h_data_vectors, 
    int** h_index, int** h_topk_index, float** h_topk_cos_dist,
    int n_query, int n_batch, int n_dim,
    int k /*查找的最近邻个数*/
){ 
    /**
    * 对一个batch的查询向量，找出余弦距离最近的topk，返回一个形状为 [batch, k] 的索引矩阵
    **/
//    table_2D("h_topk_index", h_topk_index, n_query, k);

    float alpha = 1.0f; 
    float beta = 0.0f;

    const int NUM_STREAMS = 0; // cuda流的数量
    bool query_copied = false; // query常驻显存

    dim3 queryDim(n_query);
    dim3 dataDim(n_batch);
    dim3 vectorDim(n_dim);

    cudaStream_t streams[NUM_STREAMS];
    
    size_t size_query = n_query * n_dim * sizeof(float);
    size_t size_data = n_batch * n_dim * sizeof(float);
    size_t size_dist = n_query * n_batch * sizeof(float);
    size_t size_index = n_query * n_batch * sizeof(int);
    size_t size_topk_dist = n_query * k * sizeof(float);
    size_t size_topk_idx = n_query * k * sizeof(int);

    // cuBLAS句柄
    cublasHandle_t handle;
    // cublasSetStream(handle, streams[0]); 

    // 分配设备内存
    float *d_query_vectors, *d_data_vectors, *d_inner_product, *d_topk_cos_dist,
        *d_query_norm, *d_data_norm;
    int *d_index, *d_topk_index;
    int *d_heap_mutexes, *d_heap_sizes;  /* 全局内存中的堆互斥锁和堆大小数组 */
    {
        CUDATimer timer_manage("GPU Memory Allocation", ENABLE_CUDA_TIMING);

        cudaMalloc(&d_query_vectors, size_query);
        cudaMalloc(&d_data_vectors, size_data);
        cudaMalloc(&d_inner_product, size_dist);/*存储各个query需要查找的data向量的距离*/
        cudaMalloc(&d_index, size_index);/*存储各个query需要查找的data向量的索引*/
        cudaMalloc(&d_topk_cos_dist, size_topk_dist);/*存储topk距离*/
        cudaMalloc(&d_topk_index, size_topk_idx);/*存储topk索引*/

        cudaMalloc(&d_query_norm, n_query * sizeof(float)); /*存储query的l2 Norm*/
        cudaMalloc(&d_data_norm, n_batch * sizeof(float)); /*存储data的l2 Norm*/
        
        cudaMalloc(&d_heap_mutexes, n_query * sizeof(int)); /*存储每个query的堆互斥锁*/
        cudaMalloc(&d_heap_sizes, n_query * sizeof(int)); /*存储每个query的堆大小*/

        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        }

        cublasCreate(&handle);
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
            n_batch * sizeof(int),
            h_index[0],
            n_batch * sizeof(int),
            n_batch * sizeof(int),
            n_query,
            cudaMemcpyHostToDevice
        );
        // CHECK_CUDA_ERRORS;

        /* 初始化距离数组（使用fill kernel替代thrust::fill） */
        dim3 fill_block(256);
        int fill_grid_size = (n_query * k + fill_block.x - 1) / fill_block.x;
        dim3 fill_grid(fill_grid_size);
        fill_kernel<<<fill_grid, fill_block>>>(
            d_topk_cos_dist,
            FLT_MAX,
            n_query * k
        );
        // cudaMemset((void*)d_topk_cos_dist, (int)0xEF, n_query * k * sizeof(float)) /*也可以投机取巧用memset，正好将数组为一个非常大的负数*/
        // table_cuda_2D("topk cos distance", d_topk_cos_dist, n_query, k);

        /* 初始化堆互斥锁和堆大小数组 */
        cudaMemset(d_heap_mutexes, 0, n_query * sizeof(int)); /*互斥锁初始化为0（未锁定）*/
        cudaMemset(d_heap_sizes, 0, n_query * sizeof(int)); /*堆大小初始化为0（空堆）*/

    }

    // print_cuda_2D("index matrix", d_index, n_query, n_batch);
    // print_cuda_2D("cos distance matrix", d_inner_product, n_query, n_batch);
    // print_2D("query vector", h_query_vectors, n_query, n_dim);
    // print_2D("data vector", h_data_vectors, n_batch, n_dim);
    // print_cuda_2D("query vector", d_query_vectors, n_query, n_dim);
    // print_cuda_2D("data vector", d_data_vectors, n_batch, n_dim);
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

        // table_cuda_2D("topk cos distance", d_topk_cos_dist, n_query, k);
       
        // table_cuda_1D("query_norm", d_query_norm, n_query);
        // table_cuda_1D("data_norm", d_data_norm, n_batch);
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
            d_inner_product, n_batch
        );    
        
        cudaDeviceSynchronize(); 

        // print_cuda_2D("inner product", d_inner_product, n_query, n_batch);

        // table_cuda_2D("topk index", d_topk_index, n_query, k);
        // table_cuda_2D("topk cos distance", d_topk_cos_dist, n_query, k);

        fusion_cos_topk_kernel<<<queryDim, dataDim>>>(
            d_query_norm, d_data_norm, d_inner_product, d_index,
            d_topk_index, d_topk_cos_dist,
            d_heap_mutexes, d_heap_sizes,
            n_query, n_batch, k
        );

        // cos_distance_kernel<<<dataDim, queryDim>>>(
            // d_query_norm, d_data_norm, d_inner_product,
            // n_query, n_batch, n_dim
        // );
        // table_cuda_2D("topk index", d_topk_index, n_query, k);
        // table_cuda_2D("topk cos distance", d_topk_cos_dist, n_query, k);

        cudaDeviceSynchronize(); 
        
        // 设置CUDA设备以确保RAFT库使用正确的设备
        

    }


    {
        CUDATimer timer_trans2("D2H Data Transfer", ENABLE_CUDA_TIMING);
        cudaMemcpy(h_topk_index[0], d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost);        
        cudaMemcpy(h_topk_cos_dist[0], d_topk_cos_dist, n_query * k * sizeof(float), cudaMemcpyDeviceToHost);        
    }

    {
        CUDATimer timer_manage2("GPU Memory Free", ENABLE_CUDA_TIMING, false);
        cublasDestroy(handle);
        cudaFree(d_query_vectors);
        cudaFree(d_data_vectors);
        cudaFree(d_inner_product);
        cudaFree(d_query_norm);
        cudaFree(d_data_norm);
        cudaFree(d_index);
        cudaFree(d_topk_cos_dist);
        cudaFree(d_topk_index);
        cudaFree(d_heap_mutexes);
        cudaFree(d_heap_sizes);
        
        // 销毁CUDA流
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }

    // CHECK_CUDA_ERRORS;
}