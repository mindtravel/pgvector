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
#include <cfloat>
#include "../utils.cuh"

#include "fusion_cos_topk.cuh"
#include "../unit_tests/common/test_utils.cuh"
#include "pch.h"
#include "warpsortfilter/warpsort_utils.cuh"
#include "warpsortfilter/warpsort.cuh"
#include "../utils.cuh"

#define ENABLE_CUDA_TIMING 0

namespace pgvector {
namespace fusion_cos_topk_warpsort {

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
__global__ void fusion_cos_topk_warpsort_kernel(
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
    
    float query_norm = d_query_norm[row];
    
    /* 获取dummy值，用于无效元素（确保所有线程都参与同步）*/
    /* 使用 WarpSort 基类的静态方法获取正确的dummy值 */
    using BaseWarpSort = WarpSort<Capacity, Ascending, T, IdxT>;
    const T dummy_val = BaseWarpSort::kDummy();

    /* 按照 laneId 访问数据 */
    const T* row_inner_product = d_inner_product + row * len;
    const IdxT* row_index = d_index + row * len;
    
    /* 
     * 关键修复：使用固定次数的循环，确保所有线程执行相同次数的迭代
     * 这对于 WarpSortFiltered 的 any() 和 __any_sync() 同步是必需的
     * 
     * 问题：原来的条件循环 `for (int i = ...; i < len; i += ...)` 导致不同线程
     * 执行不同次数的迭代，使得 queue.add() 的调用不同步，导致 __any_sync() 死锁
     * 
     * 解决方案：计算最大迭代次数，所有线程执行相同次数的循环，
     * 每个迭代都同步调用 queue.add()（无论是否有有效数据）
     */
    
    /* 确保所有线程都执行到这里 */
    __syncwarp();
    
    /* 计算最大迭代次数：ceil(len / (n_warps * kWarpSize)) */
    int max_iter = (len + n_warps * kWarpSize - 1) / (n_warps * kWarpSize);
    
    for (int iter = 0; iter < max_iter; iter++) {
        /* 在每个迭代开始时同步 */
        __syncwarp();
        
        /* 计算当前迭代处理的索引 */
        int i = warp_id * kWarpSize + lane + iter * n_warps * kWarpSize;
        bool has_data = (i < len);
        
        if (has_data) {
            float data_norm = d_data_norm[i];
            float inner_product = row_inner_product[i];
            IdxT index = row_index[i];
            
            /* 边界检查：避免无效数据 */
            /* 对于索引类型，如果为负数（有符号）或特殊标记值，视为无效 */
            bool is_valid_index = true;
            if constexpr (std::is_signed_v<IdxT>) {
                is_valid_index = (index >= 0);
            }
            
            if (data_norm >= 1e-6f && is_valid_index) {
                float cos_similarity = inner_product / (query_norm * data_norm);
                float cos_distance = 1.0f - cos_similarity;
                queue.add(cos_distance, index);
            } else {
                /* 无效数据，添加dummy值 */
                queue.add(dummy_val, IdxT{});
            }
        } else {
            /* 超出范围，添加dummy值以确保所有线程同步 */
            queue.add(dummy_val, IdxT{});
        }
    }
    
    /* 确保所有线程都完成了循环 */
    __syncwarp();
    
    /* 把 buffer 中剩余数合并到 queue 中 */
    queue.done();
    
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
cudaError_t fusion_cos_topk_warpsort(
    const T* d_query_norm, const T* d_data_norm, const T* d_inner_product, const IdxT* d_index,
    int batch_size, int len, int k,
    T* output_vals, IdxT* output_idx,
    bool select_min,
    cudaStream_t stream  // 默认参数在头文件中声明，这里不重复
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
            fusion_cos_topk_warpsort_kernel<64, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        } else if (capacity <= 64) {
            fusion_cos_topk_warpsort_kernel<128, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        } else {
            fusion_cos_topk_warpsort_kernel<256, true, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        }
    } else {
        if (capacity <= 32) {
            fusion_cos_topk_warpsort_kernel<64, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index,
                batch_size, len, k, 
                output_vals, output_idx
            );
        } else if (capacity <= 64) {
            fusion_cos_topk_warpsort_kernel<128, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        } else {
            fusion_cos_topk_warpsort_kernel<256, false, T, IdxT><<<grid, block, 0, stream>>>(
                d_query_norm, d_data_norm, d_inner_product, d_index, 
                batch_size, len, k, 
                output_vals, output_idx
            );
        }
    }
    
    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t fusion_cos_topk_warpsort<float, int>(
    const float*, const float*, const float*, const int*, int, int, int, float*, int*, bool, cudaStream_t);

template cudaError_t fusion_cos_topk_warpsort<float, uint32_t>(
    const float*, const float*, const float*, const uint32_t*, int, int, int, float*, uint32_t*, bool, cudaStream_t);

} // namespace warpsort
} // namespace pgvector


void cuda_cos_topk_warpsort(
    float** h_query_vectors, 
    float** h_data_vectors, 
    int** h_topk_index, 
    float** h_topk_cos_dist,
    int n_query, 
    int n_batch, 
    int n_dim,
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

        /* 使用 CUDA kernel 并行生成顺序索引 [0, 1, 2, ..., n_batch-1] */
        // 线程模型：每个block处理一个query，每个block使用256个线程（或更少）
        const int threads_per_block = 256;
        dim3 block_dim((n_batch < threads_per_block) ? n_batch : threads_per_block);
        dim3 grid_dim(n_query);
        
        generate_sequence_indices_kernel<<<grid_dim, block_dim>>>(
            d_index, n_query, n_batch);
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


        // COUT_ENDL("finish data transfer");


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
        // COUT_ENDL("finish matrix multiply");

        // print_cuda_2D("inner product", d_inner_product, n_query, n_batch);

        // table_cuda_2D("topk index", d_topk_index, n_query, k);
        // table_cuda_2D("topk cos distance", d_topk_cos_dist, n_query, k);
    }

    {
        CUDATimer timer_compute("Kernel Execution: cos + topk", ENABLE_CUDA_TIMING);

        pgvector::fusion_cos_topk_warpsort::fusion_cos_topk_warpsort(
            d_query_norm, d_data_norm, d_inner_product, d_index,
            n_query, n_batch, k,
            d_topk_cos_dist, d_topk_index,
            true /* select min */
        );

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
        // 销毁CUDA流
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }

    // CHECK_CUDA_ERRORS;
}