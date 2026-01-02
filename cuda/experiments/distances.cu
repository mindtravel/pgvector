#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include "../l2norm/l2norm.cuh"
#include "distances.h"
#include "pch.h"

#include "../unit_tests/common/test_utils.cuh"


#define ENABLE_CUDA_TIMING 1 /*是否启用CUDATimer计时*/

__global__ void cos_distance_kernel(
    float *query_l2_norm, float *data_l2_norm, float *d_cos_dist, 
    int n_query, int n_batch, int n_dim
) {
    int data_id = blockIdx.x;
    int query_id = threadIdx.x;
    // int idx = data_id * n_query + query_id;
    int idx = data_id + query_id * n_batch;

    // 边界检查 + 错误防护
    if (query_id >= n_query || data_id >= n_batch || idx >= n_batch * n_query) {
        #ifdef DEBUG
        printf("Boundary Error: query_id=%d/%d, data_id=%d/%d\n", 
               query_id, n_query, data_id, n_batch);
        #endif
        return;
    }

    // // 共享内存缓存query范数
    // __shared__ float s_query_norm[256];
    // if (threadIdx.y == 0 && threadIdx.x < n_query) {
    //     s_query_norm[threadIdx.x] = query_l2_norm[threadIdx.x];
    // }
    // __syncthreads();

    // float query_norm = s_query_norm[query_id % 256]; // 假设BLOCK_SIZE=256
    float query_norm = query_l2_norm[query_id]; // 假设BLOCK_SIZE=256
    float data_norm = data_l2_norm[data_id];

    // 范数合法性检查
    if (!isfinite(query_norm) || !isfinite(data_norm) || 
        fabsf(query_norm) < 1e-6f || fabsf(data_norm) < 1e-6f) {
        d_cos_dist[idx] = 0.0f;
        return;
    }

    // 计算归一化余弦距离
    d_cos_dist[idx] /= (query_norm * data_norm);
    // d_cos_dist[idx] = (query_norm);
}

/**
 * 计算一个batch的query和待检索的list之间的余弦距离 [n_querys, n_dim] * [n_batch * n_dim].T
 */
void cuda_cosine_dist(const float** h_query_vectors, const float** h_data_vectors, float** h_cos_dist,
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