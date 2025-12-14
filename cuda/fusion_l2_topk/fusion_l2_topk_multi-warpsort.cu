/**
 * Warp-Sort Top-K Implementation for pgvector
 * 
 * Based on RAFT (RAPIDS AI) warp-sort implementation:
 * - raft/cpp/include/raft/matrix/detail/select_warpsort.cuh
 * - raft/cpp/include/raft/util/bitonic_sort.cuh
 * 
 * This implementation provides GPU-accelerated top-k selection using
 * warp-level primitives and bitonic sorting networks.
 * 
 * Key features:
 * - Support for k up to 256 (kMaxCapacity)
 * - Warp-level parallelism using shuffle operations
 * - Register-based storage for minimal memory overhead
 * - Bitonic merge network for efficient sorting
 * 
 * Copyright (c) 2024, pgvector
 * Adapted from RAFT (Apache 2.0 License)
 */

#include <limits>
#include <type_traits>
#include <math_constants.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include "fusion_l2_topk.cuh"
#include "../unit_tests/common/test_utils.cuh"
#include "pch.h"
#include "warpsortfilter/warpsort_utils.cuh"
#include "warpsortfilter/warpsort.cuh"

#define ENABLE_CUDA_TIMING 0

namespace pgvector {
namespace fusion_l2_topk_warpsort {

using namespace warpsort_utils;
using namespace warpsort;

// ============================================================================
// Public API: Top-K Selection Kernel
// ============================================================================

/**
 * 从矩阵每一行中选取 top-k 最小或最大元素。
 * 
 * 每个 CUDA block 处理一行。每个 block 内的 warp 独立地使用 WarpSortFiltered 算法完成排序筛选。
 * 
 * @param[in] d_query_norm        输入的query L2范数内积矩阵，形状为 [n_query]
 * @param[in] d_data_norm        输入的data L2范数内积矩阵，形状为 [n_batch]
 * @param[in] d_inner_product        输入的内积矩阵，形状为 [n_query, n_batch]
 * @param[in] d_index        输入的索引，形状为 [n_query, n_batch]
 * @param[in] batch_size   行数（批大小）
 * @param[in] len          每行的元素个数
 * @param[in] k            选取的元素个数
 * @param[out] output_vals 输出 top-k 值，形状为 [n_query, k]
 * @param[out] output_idx  输出 top-k 对应的索引，形状为 [n_query, k]
 * @param[in] select_min   若为 true 选取最小的 k 个，否则选取最大的 k 个
 */
template<int Capacity, bool Ascending, typename T, typename IdxT>
__global__ void fusion_l2_topk_warpsort_kernel(
    const T* __restrict__ d_query_norm,
    const T* __restrict__ d_data_norm,
    const T* __restrict__ d_inner_product,
    const IdxT* __restrict__ d_index,
    int batch_size,
    int len,
    int k,
    T* __restrict__ output_vals,
    IdxT* __restrict__ output_idx)
{
    const int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = laneId();
    const int n_warps = blockDim.x / kWarpSize; /* 一共参与WarpSort的数量（目前都是1） */
    
    /* 在一个warp中，建立一个长度为k的队列 */
    WarpSortFiltered<Capacity, Ascending, T, IdxT> queue(k); 

    //multi_warp
    __shared__ T shared_vals[8*k];
    __shared__ IdxT shared_idx[8*k];
    
    float query_norm = d_query_norm[row];

    /* 按照 laneId 访问数据 */
    const T* row_inner_product = d_inner_product + row * len;
    const IdxT* row_index = d_index + row * len;
    for (int i = warp_id * kWarpSize + lane; i < len; i += n_warps * kWarpSize) {
        float data_norm = d_data_norm[i];
        float inner_product = row_inner_product[i];
        IdxT index = row_index[i];  /* 修复：使用正确的索引值 */
        float l2_similarity = query_norm*query_norm + data_norm*data_norm - 2.0f * inner_product;
        queue.add(l2_distance, index);
    }
    
    /* 把 buffer 中剩余数合并到 queue 中 */
    queue.done();
        
    if(lane < k){
        shared_vals[warp_id * k + lane] = queue.vals()[lane];
        shared_idx[warp_id * k + lane] = queue.idxs()[lane];
    }
    
    /* 将 queue 中的数存储到显存中（所有线程都要调用）*/
    if (warp_id == 0) {
        T* row_out_val = output_vals + row * k;
        IdxT* row_out_idx = output_idx + row * k;
        queue.store(row_out_val, row_out_idx);
    }
}

/**
 * Host function to launch top-k selection.
 * Automatically chooses appropriate capacity based on k.
 */
template<typename T, typename IdxT>
cudaError_t fusion_l2_topk_warpsort(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product, const IdxT* d_index,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream = 0
)
{
    if (k > kMaxCapacity) {
        return cudaErrorInvalidValue;
    }
    
    /* 
     * 选择合适的 Capacity
     * 
     * WarpSortFiltered 需要 buffer 空间，因此：
     * - Capacity 必须 > k（不能等于 k）
     * - 最小使用 64（确保 kMaxArrLen >= 2）
     * - 选择最小的满足 Capacity > k 的 2 的幂
     */
    int capacity = 32;  /* 最小使用 32 */
    while (capacity < k) capacity <<= 1;  /* 注意：必须 > k，不能等于 */
    
    dim3 block(32);  /* 使用32线程（单个warp）*/
    dim3 grid(batch_size);
    
    /* 模板的非类型参数必须是常量，所以只能用这一系列分支来使用不同尺寸的函数 */
    if (select_min) {
        if (capacity <= 32) {
            fusion_l2_topk_warpsort_kernel<64, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        } else if (capacity <= 64) {
            fusion_l2_topk_warpsort_kernel<128, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        } else {
            fusion_l2_topk_warpsort_kernel<256, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        }
    } else {
        if (capacity <= 32) {
            fusion_l2_topk_warpsort_kernel<64, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index,
                batch_size, len, k, 
                output_vals, output_idx
            );
        } else if (capacity <= 64) {
            fusion_l2_topk_warpsort_kernel<128, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        } else {
            fusion_l2_topk_warpsort_kernel<256, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        }
    }
    
    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t fusion_l2_topk_warpsort<float, int>(
    const float*, const float*, const float*, const int*, int, int, int, float*, int*, bool, cudaStream_t);

template cudaError_t fusion_l2_topk_warpsort<float, uint32_t>(
    const float*, const float*, const float*, const uint32_t*, int, int, int, float*, uint32_t*, bool, cudaStream_t);

} // namespace warpsort
} // namespace pgvector


void cuda_l2_topk_warpsort(
    float** h_query_vectors, float** h_data_vectors, 
    int** h_index, int** h_topk_index, float** h_topk_l2_dist,
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
    float *d_query_vectors, *d_data_vectors, *d_inner_product, *d_topk_l2_dist,
        *d_query_norm, *d_data_norm;
    int *d_index, *d_topk_index;
    {
        CUDATimer timer_manage("GPU Memory Allocation", ENABLE_CUDA_TIMING);

        cudaMalloc(&d_query_vectors, size_query);
        cudaMalloc(&d_data_vectors, size_data);
        cudaMalloc(&d_inner_product, size_dist);/*存储各个query需要查找的data向量的距离*/
        cudaMalloc(&d_index, size_index);/*存储各个query需要查找的data向量的索引*/
        cudaMalloc(&d_topk_l2_dist, size_topk_dist);/*存储topk距离*/
        cudaMalloc(&d_topk_index, size_topk_idx);/*存储topk索引*/

        cudaMalloc(&d_query_norm, n_query * sizeof(float)); /*存储query的l2 Norm*/
        cudaMalloc(&d_data_norm, n_batch * sizeof(float)); /*存储data的l2 Norm*/

        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        }

        cublasCreate(&handle);
    }

    // 复制数据到设备
    {
        // COUT_ENDL("begin data transfer");

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

        /* 初始化距离数组（为一个小于-1的负数） */
        thrust::fill(
            thrust::device_pointer_cast(d_topk_l2_dist),/*使用pointer_cast不用创建临时对象*/
            thrust::device_pointer_cast(d_topk_l2_dist) + (n_query * k),  /* 使用元素数量而非字节数 */
            FLT_MAX
        );
        // cudaMemset((void*)d_topk_l2_dist, (int)0xEF, n_query * k * sizeof(float)) /*也可以投机取巧用memset，正好将数组为一个非常大的负数*/
        // table_cuda_2D("topk l2 distance", d_topk_l2_dist, n_query, k);


        // COUT_ENDL("finish data transfer");


    }

    // print_cuda_2D("index matrix", d_index, n_query, n_batch);
    // print_cuda_2D("l2 distance matrix", d_inner_product, n_query, n_batch);
    // print_2D("query vector", h_query_vectors, n_query, n_dim);
    // print_2D("data vector", h_data_vectors, n_batch, n_dim);
    // print_cuda_2D("query vector", d_query_vectors, n_query, n_dim);
    // print_cuda_2D("data vector", d_data_vectors, n_batch, n_dim);
    // print_cuda_2D("topk index matrix", d_topk_index, n_query, k);
    // print_cuda_2D("topk l2 distance matrix", d_topk_l2_dist, n_query, k);

    /* 核函数执行 */
    {
        // COUT_ENDL("begin_kernel");

        CUDATimer timer_compute("Kernel Execution: l2 Norm + matrix multiply", ENABLE_CUDA_TIMING);

        l2_norm_kernel<<<queryDim, vectorDim, n_dim * sizeof(float)>>>(
            d_query_vectors, d_query_norm, 
            n_query, n_dim
        );

        l2_norm_kernel<<<dataDim, vectorDim, n_dim * sizeof(float)>>>(
            d_data_vectors, d_data_norm, 
            n_batch, n_dim
        );        
        // COUT_ENDL("finish l2 norm");

        // table_cuda_2D("topk l2 distance", d_topk_l2_dist, n_query, k);
       
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
        // COUT_ENDL("finish matrix multiply");

        // print_cuda_2D("inner product", d_inner_product, n_query, n_batch);

        // table_cuda_2D("topk index", d_topk_index, n_query, k);
        // table_cuda_2D("topk l2 distance", d_topk_l2_dist, n_query, k);
    }

    {
        CUDATimer timer_compute("Kernel Execution: l2 + topk", ENABLE_CUDA_TIMING);

        pgvector::fusion_l2_topk_warpsort::fusion_l2_topk_warpsort(
            d_query_norm, d_data_norm, d_inner_product, d_index,
            n_query, n_batch, k,
            d_topk_l2_dist, d_topk_index,
            true /* select min */
        );

        // table_cuda_2D("topk index", d_topk_index, n_query, k);
        // table_cuda_2D("topk l2 distance", d_topk_l2_dist, n_query, k);

        cudaDeviceSynchronize(); 
        
        // 设置CUDA设备以确保RAFT库使用正确的设备
        

    }


    {
        CUDATimer timer_trans2("D2H Data Transfer", ENABLE_CUDA_TIMING);
        cudaMemcpy(h_topk_index[0], d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost);        
        cudaMemcpy(h_topk_l2_dist[0], d_topk_l2_dist, n_query * k * sizeof(float), cudaMemcpyDeviceToHost);        
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
        cudaFree(d_topk_l2_dist);
        cudaFree(d_topk_index);
        // 销毁CUDA流
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }

    // CHECK_CUDA_ERRORS;
}