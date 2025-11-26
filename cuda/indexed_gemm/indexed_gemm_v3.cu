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

template<int Tile>
__device__ __forceinline__ void load_tile_vec4(const float4* lhs_vec4,
                                               const float4* rhs_vec4,
                                               int base_idx,
                                               int vec4_count,
                                               float4 (&lhs_tile)[Tile],
                                               float4 (&rhs_tile)[Tile]) {
    #pragma unroll
    for (int t = 0; t < Tile; ++t) {
        int idx = base_idx + t;
        if (idx < vec4_count) {
            lhs_tile[t] = lhs_vec4[idx];
            rhs_tile[t] = rhs_vec4[idx];
        } else {
            lhs_tile[t] = make_float4(0.f, 0.f, 0.f, 0.f);
            rhs_tile[t] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
}

template<int Tile>
__device__ __forceinline__ float accumulate_tile(const float4 (&lhs_tile)[Tile],
                                                 const float4 (&rhs_tile)[Tile],
                                                 int valid_count,
                                                 float sum) {
    #pragma unroll
    for (int t = 0; t < Tile; ++t) {
        if (t < valid_count) {
            const float4& l = lhs_tile[t];
            const float4& r = rhs_tile[t];
            sum = fmaf(l.x, r.x, sum);
            sum = fmaf(l.y, r.y, sum);
            sum = fmaf(l.z, r.z, sum);
            sum = fmaf(l.w, r.w, sum);
        }
    }
    return sum;
}

template<int Dim>
__device__ __forceinline__ float dot_product_tiled(const float* __restrict__ lhs,
                                                   const float* __restrict__ rhs) {
    constexpr int kVec4Count = Dim / 4;
    constexpr int kTile = 4;
    if constexpr (kVec4Count == 0) {
        return 0.0f;
    } else {
        constexpr int tile_count = (kVec4Count + kTile - 1) / kTile;
        const float4* lhs_vec4 = reinterpret_cast<const float4*>(lhs);
        const float4* rhs_vec4 = reinterpret_cast<const float4*>(rhs);

        float4 cur_lhs[kTile];
        float4 cur_rhs[kTile];
        load_tile_vec4<kTile>(lhs_vec4, rhs_vec4, 0, kVec4Count, cur_lhs, cur_rhs);

        float sum = 0.0f;

        if constexpr (tile_count == 1) {
            sum = accumulate_tile<kTile>(cur_lhs, cur_rhs, kVec4Count, sum);
        } else {
            float4 next_lhs[kTile];
            float4 next_rhs[kTile];

            #pragma unroll
            for (int tile = 0; tile < tile_count; ++tile) {
                int next_base = (tile + 1) * kTile;
                if (tile + 1 < tile_count) {
                    load_tile_vec4<kTile>(lhs_vec4, rhs_vec4, next_base, kVec4Count, next_lhs, next_rhs);
                }

                int valid = kVec4Count - tile * kTile;
                valid = valid > kTile ? kTile : valid;
                sum   = accumulate_tile<kTile>(cur_lhs, cur_rhs, valid, sum);

                if (tile + 1 < tile_count) {
                    #pragma unroll
                    for (int t = 0; t < kTile; ++t) {
                        cur_lhs[t] = next_lhs[t];
                        cur_rhs[t] = next_rhs[t];
                    }
                }
            }
        }

        return sum;
    }
}

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

// 显式实例化：v3版本（一个block处理8个query）
// Capacity 64, QueriesPerBlock=8
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<64, true, 96, 4>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<64, true, 128, 4>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<64, true, 200, 4>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
// Capacity 128, QueriesPerBlock=8
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<128, true, 96, 4>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<128, true, 128, 4>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<128, true, 200, 4>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
// Capacity 256, QueriesPerBlock=8
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<256, true, 96, 4>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<256, true, 128, 4>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<256, true, 200, 4>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);

// Generic版本，QueriesPerBlock=8
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

// 显式实例化：v3_32版本（一个block处理16个query）
// Capacity 64, QueriesPerBlock=16
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<64, true, 96, 16>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<64, true, 128, 16>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<64, true, 200, 16>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
// Capacity 128, QueriesPerBlock=16
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<128, true, 96, 16>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<128, true, 128, 16>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<128, true, 200, 16>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
// Capacity 256, QueriesPerBlock=16
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<256, true, 96, 16>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<256, true, 128, 16>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);
 
template __global__ void indexed_inner_product_with_topk_kernel_v3_static<256, true, 200, 16>(
    float* __restrict__, float* __restrict__, int* __restrict__,
    float* __restrict__, float* __restrict__,
    int, int, int,
    float* __restrict__, int* __restrict__);

// Generic版本，QueriesPerBlock=16
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



