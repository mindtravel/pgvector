#include "indexed_gemm.cuh"
#include "../pch.h"
#include "../warpsortfilter/warpsort_utils.cuh"
#include "../warpsortfilter/warpsort_topk.cu"
#include "../unit_tests/common/test_utils.cuh"
#include <cublas_v2.h>
#include <algorithm>

using namespace pgvector::warpsort_utils;
using namespace pgvector::warpsort;
using namespace pgvector::warpsort_topk;

/**
 * 提取选中的query的kernel
 */
__global__ void extract_selected_query_kernel(
    const float* __restrict__ d_query_group,
    const int* __restrict__ d_query_index,
    float* __restrict__ d_selected_query,
    int n_selected_querys,
    int n_dim
) {
    const int query_id = blockIdx.x;
    const int dim_id = threadIdx.x;
    
    if (query_id >= n_selected_querys || dim_id >= n_dim) return;
    
    const int query_global_id = d_query_index[query_id];
    d_selected_query[query_id * n_dim + dim_id] = 
        d_query_group[query_global_id * n_dim + dim_id];
}

/**
 * 提取选中的query norm的kernel
 */
__global__ void extract_selected_query_norm_kernel(
    const float* __restrict__ d_query_norm,
    const int* __restrict__ d_query_index,
    float* __restrict__ d_selected_query_norm,
    int n_selected_querys
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_selected_querys) return;
    
    const int query_global_id = d_query_index[idx];
    d_selected_query_norm[idx] = d_query_norm[query_global_id];
}

/**
 * 将内积矩阵转换为余弦距离矩阵的kernel（同时处理列主序到行主序的转换）
 * 
 * 输入：内积矩阵 [n_query, n_vectors] (列主序，来自cuBLAS)
 * 输出：余弦距离矩阵 [n_query, n_vectors] (行主序，供select_k使用)
 * 
 * 公式：cos_distance = 1 - cos_similarity = 1 - dot_product / (query_norm * vector_norm)
 * 
 * 注意：cuBLAS返回列主序，所以输入访问：d_inner_product[query_id + vec_id * n_query]
 *      输出需要行主序，所以输出访问：d_cos_distance[query_id * n_vectors + vec_id]
 */
__global__ void inner_product_to_cos_distance_kernel(
    const float* __restrict__ d_inner_product,  // [n_query, n_vectors] (列主序)
    const float* __restrict__ d_query_norm,     // [n_query]
    const float* __restrict__ d_vector_norm,    // [n_vectors]
    float* __restrict__ d_cos_distance,        // [n_query, n_vectors] (行主序)
    int n_query,
    int n_vectors
) {
    const int query_id = blockIdx.x;
    const int vec_id = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (query_id >= n_query || vec_id >= n_vectors) return;
    
    // 列主序输入：d_inner_product[query_id + vec_id * n_query]
    const float dot_product = d_inner_product[query_id + vec_id * n_query];
    const float query_norm = d_query_norm[query_id];
    const float vector_norm = d_vector_norm[vec_id];
    
    // 行主序输出：d_cos_distance[query_id * n_vectors + vec_id]
    if (query_norm < 1e-6f || vector_norm < 1e-6f) {
        d_cos_distance[query_id * n_vectors + vec_id] = 2.0f;  // 最大距离
    } else {
        const float cos_similarity = dot_product / (query_norm * vector_norm);
        d_cos_distance[query_id * n_vectors + vec_id] = 1.0f - cos_similarity;
    }
}

/**
 * 使用cublas_gemm计算内积矩阵，然后选择top-k
 * 
 * 适用于大规模query场景（n_query > 100 && n_vectors > 512）
 */
void indexed_inner_product_with_topk_gemm(
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_query_index,
    
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    
    int n_selected_querys,
    int n_selected_vectors,
    int n_dim,
    int k,
    
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream = 0
) {
    // 创建cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);
    if (stream != 0) {
        cublasSetStream(handle, stream);
    }
    
    // 分配临时内存：内积矩阵和余弦距离矩阵
    const size_t matrix_size = static_cast<size_t>(n_selected_querys) * n_selected_vectors * sizeof(float);
    float* d_inner_product = nullptr;
    float* d_cos_distance = nullptr;
    
    cudaMalloc(&d_inner_product, matrix_size);
    cudaMalloc(&d_cos_distance, matrix_size);
    CHECK_CUDA_ERRORS;
    
    // 准备query矩阵（需要重新排列，只包含选中的query）
    float* d_selected_query = nullptr;
    cudaMalloc(&d_selected_query, static_cast<size_t>(n_selected_querys) * n_dim * sizeof(float));
    CHECK_CUDA_ERRORS;
    
    // 准备query norm数组（只包含选中的query）
    float* d_selected_query_norm = nullptr;
    cudaMalloc(&d_selected_query_norm, n_selected_querys * sizeof(float));
    CHECK_CUDA_ERRORS;
    
    // 使用kernel提取选中的query
    {
        dim3 block(n_dim);
        dim3 grid(n_selected_querys);
        extract_selected_query_kernel<<<grid, block, 0, stream>>>(
            d_query_group,
            d_query_index,
            d_selected_query,
            n_selected_querys,
            n_dim
        );
    }
    CHECK_CUDA_ERRORS;
    
    // 使用kernel提取选中的query norm
    {
        dim3 block(256);
        dim3 grid((n_selected_querys + 255) / 256);
        extract_selected_query_norm_kernel<<<grid, block, 0, stream>>>(
            d_query_norm,
            d_query_index,
            d_selected_query_norm,
            n_selected_querys
        );
    }
    CHECK_CUDA_ERRORS;
    
    // 使用cuBLAS计算内积矩阵
    // 直接计算 C = B^T * A，得到 [n_selected_querys, n_selected_vectors] (列主序)
    // 其中：
    // B^T: selected_query转置 [n_selected_querys, n_dim]
    // A: cluster_vector [n_dim, n_selected_vectors] (列主序)
    // C: inner_product [n_selected_querys, n_selected_vectors] (列主序)
    // 这样可以直接得到我们想要的布局，无需手动转置
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // 计算 C = B^T * A，直接得到 [n_selected_querys, n_selected_vectors]
    cublasSgemm(
        handle,
        CUBLAS_OP_T,        // B转置: [n_selected_querys, n_dim]
        CUBLAS_OP_N,        // A不转置: [n_dim, n_selected_vectors]
        n_selected_querys,  // M: C的行数
        n_selected_vectors, // N: C的列数
        n_dim,              // K: B的列数/A的行数
        &alpha,
        d_selected_query,   // B: [n_dim, n_selected_querys] (列主序)
        n_dim,              // ldb
        d_cluster_vector,   // A: [n_dim, n_selected_vectors] (列主序)
        n_dim,              // lda
        &beta,
        d_inner_product,    // C: [n_selected_querys, n_selected_vectors] (列主序)
        n_selected_querys   // ldc
    );
    CHECK_CUDA_ERRORS;
    
    // 将内积转换为余弦距离
    {
        dim3 block(256);
        dim3 grid(n_selected_querys, (n_selected_vectors + 255) / 256);
        
        inner_product_to_cos_distance_kernel<<<grid, block, 0, stream>>>(
            d_inner_product,
            d_selected_query_norm,
            d_cluster_vector_norm,
            d_cos_distance,
            n_selected_querys,
            n_selected_vectors
        );
    }
    CHECK_CUDA_ERRORS;
    
    // 使用select_k选择top-k
    select_k<float, int>(
        d_cos_distance,
        n_selected_querys,
        n_selected_vectors,
        k,
        d_topk_dist,
        d_topk_index,
        true,  // select_min (最小距离)
        stream
    );
    CHECK_CUDA_ERRORS;
    
    // 清理
    cudaFree(d_inner_product);
    cudaFree(d_cos_distance);
    cudaFree(d_selected_query);
    cudaFree(d_selected_query_norm);
    cublasDestroy(handle);
    CHECK_CUDA_ERRORS;
}

// 前向声明v3版本的launch函数
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_indexed_inner_product_with_topk_kernel_v3(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_query_index,
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_selected_querys,
    int n_selected_vectors,
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream);

/**
 * 混合策略launch函数（v4版本）
 * 
 * 根据数据规模动态选择最优算法：
 * - 大规模query（n_query > 100 && n_vectors > 512）：使用cublas_gemm
 * - 小规模query：使用v3流式实现
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_indexed_inner_product_with_topk_kernel_v4(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_query_index,
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_selected_querys,
    int n_selected_vectors,
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream) {
    
    // 混合策略阈值
    constexpr int kGemmQueryThreshold = 100;
    constexpr int kGemmVectorThreshold = 512;
    
    // 判断使用哪种算法
    if (n_selected_querys > kGemmQueryThreshold && n_selected_vectors > kGemmVectorThreshold) {
        // 使用cublas_gemm方案
        indexed_inner_product_with_topk_gemm(
            d_query_group,
            d_cluster_vector,
            d_query_index,
            d_query_norm,
            d_cluster_vector_norm,
            n_selected_querys,
            n_selected_vectors,
            n_dim,
            k,
            d_topk_dist,
            d_topk_index,
            stream
        );
    } else {
        // 使用v3流式实现
        launch_indexed_inner_product_with_topk_kernel_v3<Capacity, Ascending, QueriesPerBlock>(
            block,
            n_dim,
            d_query_group,
            d_cluster_vector,
            d_query_index,
            d_query_norm,
            d_cluster_vector_norm,
            n_selected_querys,
            n_selected_vectors,
            k,
            d_topk_dist,
            d_topk_index,
            stream
        );
    }
}

// 显式实例化v4版本的launch函数
template void launch_indexed_inner_product_with_topk_kernel_v4<64, true, 8>(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_query_index,
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_selected_querys,
    int n_selected_vectors,
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream);

template void launch_indexed_inner_product_with_topk_kernel_v4<128, true, 8>(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_query_index,
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_selected_querys,
    int n_selected_vectors,
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream);

template void launch_indexed_inner_product_with_topk_kernel_v4<256, true, 8>(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_query_index,
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_selected_querys,
    int n_selected_vectors,
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream);

