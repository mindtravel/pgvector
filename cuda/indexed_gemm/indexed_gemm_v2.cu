#include "indexed_gemm.cuh"
#include "../pch.h"
#include "../warpsortfilter/warpsort_utils.cuh"
#include "../warpsortfilter/warpsort.cuh"
#include <stdint.h>

using namespace pgvector::warpsort_utils;
using namespace pgvector::warpsort;

__device__ __forceinline__ float dot_product_vec4_aligned(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    int length) {
    float sum = 0.0f;
    const int vec4_elems = length >> 2;
    const float4* lhs_vec4 = reinterpret_cast<const float4*>(lhs);
    const float4* rhs_vec4 = reinterpret_cast<const float4*>(rhs);

    #pragma unroll
    for (int v = 0; v < vec4_elems; ++v) {
        const float4 lhs_val = lhs_vec4[v];
        const float4 rhs_val = rhs_vec4[v];
        sum += lhs_val.x * rhs_val.x +
               lhs_val.y * rhs_val.y +
               lhs_val.z * rhs_val.z +
               lhs_val.w * rhs_val.w;
    }
    return sum;
}

__device__ __forceinline__ float dot_product_accumulate(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    int length) {
    float sum = 0.0f;
    int i = 0;

    while (i < length &&
           ((reinterpret_cast<uintptr_t>(lhs + i) |
             reinterpret_cast<uintptr_t>(rhs + i)) & (sizeof(float4) - 1))) {
        sum += lhs[i] * rhs[i];
        ++i;
    }

    const int remaining = length - i;
    const int vec4_elems = remaining >> 2;

    if (vec4_elems > 0) {
        const float4* lhs_vec4 = reinterpret_cast<const float4*>(lhs + i);
        const float4* rhs_vec4 = reinterpret_cast<const float4*>(rhs + i);

        #pragma unroll
        for (int v = 0; v < vec4_elems; ++v) {
            const float4 lhs_val = lhs_vec4[v];
            const float4 rhs_val = rhs_vec4[v];
            sum += lhs_val.x * rhs_val.x +
                   lhs_val.y * rhs_val.y +
                   lhs_val.z * rhs_val.z +
                   lhs_val.w * rhs_val.w;
        }
        i += vec4_elems << 2;
    }

    for (; i < length; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

__device__ __forceinline__ void prefetch_l2(const void* ptr) {
#if __CUDA_ARCH__ >= 700
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
#endif
}

/**
 * 流式内积计算 + top-k选择kernel（v2版本：优化数据上传）
 *
 * - 静态版本：Dim 编译期固定，由 host 端预先调度，不再在 device 端根据维度分支。
 * - 泛化版本：保留 runtime n_dim 逻辑，作为未覆盖静态维度时的回退实现。
 */
template<int Capacity, bool Ascending, int Dim>
__global__ void indexed_inner_product_with_topk_kernel_v2_static(
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
    const int query_id = blockIdx.y;
    const int query_global_id = d_query_index[query_id];
    if (query_id >= n_selected_querys) return;

    const int lane = laneId();

    const float query_norm = d_query_norm[query_global_id];
    if (query_norm < 1e-6f) return;
    const float* query_global_ptr = d_query_group + query_global_id * Dim;

    using WarpSortBase =
        pgvector::warpsort::WarpSort<Capacity, Ascending, float, int>;
    const float dummy_val = WarpSortBase::kDummy();

    WarpSortFiltered<Capacity, Ascending, float, int> queue(k);

    int max_iterations = (n_selected_vectors + kWarpSize - 1) / kWarpSize;

    for (int iter = 0; iter < max_iterations; ++iter) {
        int vec_idx = iter * kWarpSize + lane;
        bool has_valid_vec = (vec_idx < n_selected_vectors);

        // if (iter + 1 < max_iterations) {
        //     int next_vec_idx = (iter + 1) * kWarpSize + lane;
        //     if (next_vec_idx < n_selected_vectors) {
        //         const float* next_vec_ptr = d_cluster_vector + next_vec_idx * Dim;
        //         prefetch_l2(next_vec_ptr);
        //         prefetch_l2(d_cluster_vector_norm + next_vec_idx);
        //     }
        // }

        if (!has_valid_vec) {
            queue.add(dummy_val, -1);
        } else {
            const float* vec_ptr = d_cluster_vector + vec_idx * Dim;
            float dot_product = 0.0f;

            dot_product += dot_product_vec4_aligned(query_global_ptr, vec_ptr, Dim);

            float data_norm = d_cluster_vector_norm[vec_idx];
            if (data_norm < 1e-6f) {
                queue.add(dummy_val, -1);
            } else {
                float cos_similarity = dot_product / (query_norm * data_norm);
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

template<int Capacity, bool Ascending>
__global__ void indexed_inner_product_with_topk_kernel_v2_generic(
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
    const int query_id = blockIdx.y;
    const int query_global_id = d_query_index[query_id];
    if (query_id >= n_selected_querys) return;

    const int lane = laneId();

    const float query_norm = d_query_norm[query_global_id];
    if (query_norm < 1e-6f) return;
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

            if (iter + 1 < max_iterations) {
                int next_vec_idx = (iter + 1) * kWarpSize + lane;
                if (next_vec_idx < n_selected_vectors) {
                    const float* next_vec_ptr = d_cluster_vector + next_vec_idx * n_dim;
                    prefetch_l2(next_vec_ptr);
                    prefetch_l2(d_cluster_vector_norm + next_vec_idx);
                }
            }

            if (!has_valid_vec) {
                queue.add(dummy_val, -1);
            } else {
                const float* vec_ptr = d_cluster_vector + vec_idx * n_dim;
                float dot_product = dot_product_vec4_aligned(query_global_ptr, vec_ptr, n_dim);

                float data_norm = d_cluster_vector_norm[vec_idx];
                if (data_norm < 1e-6f) {
                    queue.add(dummy_val, -1);
                } else {
                    float cos_similarity = dot_product / (query_norm * data_norm);
                    float cos_distance = 1.0f - cos_similarity;
                    queue.add(cos_distance, vec_idx);
                }
            }
        }
    } else {
        for (int iter = 0; iter < max_iterations; ++iter) {
            int vec_idx = iter * kWarpSize + lane;
            bool has_valid_vec = (vec_idx < n_selected_vectors);

            if (iter + 1 < max_iterations) {
                int next_vec_idx = (iter + 1) * kWarpSize + lane;
                if (next_vec_idx < n_selected_vectors) {
                    const float* next_vec_ptr = d_cluster_vector + next_vec_idx * n_dim;
                    prefetch_l2(next_vec_ptr);
                    prefetch_l2(d_cluster_vector_norm + next_vec_idx);
                }
            }

            if (!has_valid_vec) {
                queue.add(dummy_val, -1);
            } else {
                const float* vec_ptr = d_cluster_vector + vec_idx * n_dim;
                float dot_product = dot_product_accumulate(query_global_ptr, vec_ptr, n_dim);

                float data_norm = d_cluster_vector_norm[vec_idx];
                if (data_norm < 1e-6f) {
                    queue.add(dummy_val, -1);
                } else {
                    float cos_similarity = dot_product / (query_norm * data_norm);
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

// 显式实例化：静态维度版本
// Capacity 64
template __global__ void indexed_inner_product_with_topk_kernel_v2_static<64, true, 96>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v2_static<64, true, 128>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v2_static<64, true, 200>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
 // Capacity 128
template __global__ void indexed_inner_product_with_topk_kernel_v2_static<128, true, 96>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v2_static<128, true, 128>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v2_static<128, true, 200>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
 // Capacity 256
template __global__ void indexed_inner_product_with_topk_kernel_v2_static<256, true, 96>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v2_static<256, true, 128>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v2_static<256, true, 200>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);

template __global__ void indexed_inner_product_with_topk_kernel_v2_generic<64, true>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int, int,
    float* __restrict__, int* __restrict__);

template __global__ void indexed_inner_product_with_topk_kernel_v2_generic<128, true>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int, int,
    float* __restrict__, int* __restrict__);

template __global__ void indexed_inner_product_with_topk_kernel_v2_generic<256, true>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int, int,
    float* __restrict__, int* __restrict__);



