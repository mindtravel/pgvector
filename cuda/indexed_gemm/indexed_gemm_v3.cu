#include "indexed_gemm.cuh"
#include "inner_product_utils.cuh"
#include "../pch.h"
#include "../warpsortfilter/warpsort_utils.cuh"
#include "../warpsortfilter/warpsort.cuh"
#include <stdint.h>

using namespace pgvector::warpsort_utils;
using namespace pgvector::warpsort;

/**
 * 流式内积计算 + top-k选择kernel（v3版本：一个block处理多个query）
 *
 * - 静态版本：Dim 编译期固定，由 host 端预先调度，不再在 device 端根据维度分支。
 * - 每个block处理多个query（由QueriesPerBlock指定），每个warp处理一个query。
 * - 提高GPU利用率，减少kernel启动开销。
 * 
 * @tparam QueriesPerBlock 每个block处理的query数量（建议为8，需要block有足够的warp）
 */
template<int Capacity, bool Ascending, int Dim, int QueriesPerBlock>
__global__ __launch_bounds__(256, 1) void indexed_inner_product_with_topk_kernel_v3_static(
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_query_index,
 
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
 
    int n_selected_querys,
    int n_selected_vectors,
    int k,
 
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index
) {
    static_assert((Dim % 4) == 0, "Dim must be multiple of 4 for aligned loads");
    static_assert(QueriesPerBlock > 0 && QueriesPerBlock <= 32, 
                  "QueriesPerBlock must be between 1 and 32");
    
    /* 共享内存缓存querynorm：一个block内的所有query共享 */
    __shared__ float s_query_norm[QueriesPerBlock];
    
    const int warp_id = threadIdx.x / kWarpSize;
    const int query_id = blockIdx.y * QueriesPerBlock + warp_id;
    const int lane = laneId();
    
    /* 边界处理：如果当前warp没有对应的query，直接返回
     * 例如：10个query，QueriesPerBlock=8时
     * - Block 0: warp 0-7 处理 query 0-7
     * - Block 1: warp 0-1 处理 query 8-9，warp 2-7 直接返回
     */
    if (query_id >= n_selected_querys) return;

    const int query_global_id = d_query_index[query_id];
    
    /* 每个warp的lane 0负责加载对应的query_norm到共享内存 */
    if (warp_id < QueriesPerBlock && lane == 0) {
        s_query_norm[warp_id] = d_query_norm[query_global_id];
        if (s_query_norm[warp_id] < 1e-6f) return;
    }

    __syncthreads();
    const float* query_global_ptr = d_query_group + query_global_id * Dim;

    using WarpSortBase =
        pgvector::warpsort::WarpSort<Capacity, Ascending, float, int>;
    const float dummy_val = WarpSortBase::kDummy();

    WarpSortFiltered<Capacity, Ascending, float, int> queue(k);

    int max_iterations = (n_selected_vectors + kWarpSize - 1) / kWarpSize;

    for (int iter = 0; iter < max_iterations; ++iter) {
        int vec_idx = iter * kWarpSize + lane;
        bool has_valid_vec = (vec_idx < n_selected_vectors);

        if (!has_valid_vec) {
            queue.add(dummy_val, -1);
        } else {
            const float* vec_ptr = d_cluster_vector + vec_idx * Dim;
            float dot_product = dot_product_tiled<Dim>(query_global_ptr, vec_ptr);

            float data_norm = d_cluster_vector_norm[vec_idx];
            if (data_norm < 1e-6f) {
                queue.add(dummy_val, -1);
            } else {
                float cos_similarity = dot_product / (s_query_norm[warp_id] * data_norm);
                float cos_distance = 1.0f - cos_similarity;
                queue.add(cos_distance, vec_idx);
            }
        }
    }

    __syncwarp();

    queue.done();

    __syncwarp();

    float* row_dist = d_topk_dist + query_id * k;
    int* row_idx = d_topk_index + query_id * k;
    queue.store(row_dist, row_idx);
}

/**
 * 流式内积计算 + top-k选择kernel（v3版本：一个block处理多个query - generic版本）
 *
 * - 泛化版本：保留 runtime n_dim 逻辑，作为未覆盖静态维度时的回退实现。
 * - 每个block处理多个query（由QueriesPerBlock指定），每个warp处理一个query。
 * - 提高GPU利用率，减少kernel启动开销。
 * 
 * @tparam QueriesPerBlock 每个block处理的query数量（建议为8，需要block有足够的warp）
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
__global__ __launch_bounds__(256, 1) void indexed_inner_product_with_topk_kernel_v3_generic(
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
    int* __restrict__ d_topk_index
) {
    static_assert(QueriesPerBlock > 0 && QueriesPerBlock <= 32, 
                  "QueriesPerBlock must be between 1 and 32");
    
    /* 共享内存缓存querynorm：一个block内的所有query共享 */
    __shared__ float s_query_norm[QueriesPerBlock];
    
    const int warp_id = threadIdx.x / kWarpSize;
    const int query_id = blockIdx.y * QueriesPerBlock + warp_id;
    const int lane = laneId();
    
    
    /* 边界处理：如果当前warp没有对应的query，直接返回
     * 例如：10个query，QueriesPerBlock=8时
     * - Block 0: warp 0-7 处理 query 0-7
     * - Block 1: warp 0-1 处理 query 8-9，warp 2-7 直接返回
     */
    if (query_id >= n_selected_querys) return;
    const int query_global_id = d_query_index[query_id];
    
    /* 每个warp的lane 0负责加载对应的query_norm到共享内存 */
    if (warp_id < QueriesPerBlock && lane == 0) {
        s_query_norm[warp_id] = d_query_norm[query_global_id];
        if (s_query_norm[warp_id] < 1e-6f) return;
    }
    
    /* 同步所有warp，确保共享内存已加载完成 */
    __syncthreads();

    const float* query_global_ptr = d_query_group + query_global_id * n_dim;
    const bool query_ptr_aligned =
        (reinterpret_cast<uintptr_t>(query_global_ptr) & (sizeof(float4) - 1)) == 0;
    const bool data_ptr_aligned =
        (reinterpret_cast<uintptr_t>(d_cluster_vector) & (sizeof(float4) - 1)) == 0;
    const bool prefer_vec4 = query_ptr_aligned && data_ptr_aligned && ((n_dim & 3) == 0);

    using WarpSortBase =
        pgvector::warpsort::WarpSort<Capacity, Ascending, float, int>;
    const float dummy_val = WarpSortBase::kDummy();

    WarpSortFiltered<Capacity, Ascending, float, int> queue(k);

    __syncwarp();

    int max_iterations = (n_selected_vectors + kWarpSize - 1) / kWarpSize;
    
    if (prefer_vec4) {
        for (int iter = 0; iter < max_iterations; ++iter) {
            int vec_idx = iter * kWarpSize + lane;
            bool has_valid_vec = (vec_idx < n_selected_vectors);

            if (!has_valid_vec) {
                queue.add(dummy_val, -1);
            } else {
                const float* vec_ptr = d_cluster_vector + vec_idx * n_dim;
                float dot_product = dot_product_vec4_aligned(query_global_ptr, vec_ptr, n_dim);

                float data_norm = d_cluster_vector_norm[vec_idx];
                if (data_norm < 1e-6f) {
                    queue.add(dummy_val, -1);
                } else {
                    float cos_similarity = dot_product / (s_query_norm[warp_id] * data_norm);
                    float cos_distance = 1.0f - cos_similarity;
                    queue.add(cos_distance, vec_idx);
                }
            }
        }
    } else {
        for (int iter = 0; iter < max_iterations; ++iter) {
            int vec_idx = iter * kWarpSize + lane;
            bool has_valid_vec = (vec_idx < n_selected_vectors);

            if (!has_valid_vec) {
                queue.add(dummy_val, -1);
            } else {
                const float* vec_ptr = d_cluster_vector + vec_idx * n_dim;
                float dot_product = dot_product_accumulate(query_global_ptr, vec_ptr, n_dim);

                float data_norm = d_cluster_vector_norm[vec_idx];
                if (data_norm < 1e-6f) {
                    queue.add(dummy_val, -1);
                } else {
                    float cos_similarity = dot_product / (s_query_norm[warp_id] * data_norm);
                    float cos_distance = 1.0f - cos_similarity;
                    queue.add(cos_distance, vec_idx);
                }
            }
        }
    }

    __syncwarp();

    queue.done();

    __syncwarp();

    float* row_dist = d_topk_dist + query_id * k;
    int* row_idx = d_topk_index + query_id * k;
    queue.store(row_dist, row_idx);
}

// 显式实例化（已禁用，只使用v3_fixed_probe版本）
/*
// QueriesPerBlock=4
template __global__ void indexed_inner_product_with_topk_kernel_v3_generic<64, true, 4>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int, int,
    float* __restrict__, int* __restrict__);

template __global__ void indexed_inner_product_with_topk_kernel_v3_generic<128, true, 4>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int, int,
    float* __restrict__, int* __restrict__);

template __global__ void indexed_inner_product_with_topk_kernel_v3_generic<256, true, 4>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int, int,
    float* __restrict__, int* __restrict__);

// QueriesPerBlock=16
template __global__ void indexed_inner_product_with_topk_kernel_v3_generic<64, true, 16>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int, int,
    float* __restrict__, int* __restrict__);

template __global__ void indexed_inner_product_with_topk_kernel_v3_generic<128, true, 16>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int, int,
    float* __restrict__, int* __restrict__);

template __global__ void indexed_inner_product_with_topk_kernel_v3_generic<256, true, 16>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int, int,
    float* __restrict__, int* __restrict__);
*/



