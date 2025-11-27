#include "indexed_gemm.cuh"
#include "../pch.h"
#include "../warpsortfilter/warpsort_utils.cuh"
#include "../warpsortfilter/warpsort.cuh"
#include <cfloat>
#include <stdint.h>
#include <vector>
#include <algorithm>

using namespace pgvector::warpsort_utils;
using namespace pgvector::warpsort;

// 从 indexed_gemm_v3.cu 复制的辅助函数（必须在 kernel 之前定义）
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

/**
 * Generic 版本（支持运行时维度）
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
__global__ __launch_bounds__(256, 1) 
void indexed_inner_product_with_topk_kernel_v3_fixed_probe_generic(
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_probe_queries,
    int* __restrict__ d_probe_query_offsets,
    int* __restrict__ d_probe_query_probe_indices,
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_total_clusters,  // 总的cluster数量（用于检查probe_id）
    int n_probes,  // 每个query的probe数量（用于检查probe_index_in_query）
    int max_queries_per_probe,
    int n_dim,
    int k,
    float* __restrict__ d_topk_dist,  // [n_query][n_probes][k]
    int* __restrict__ d_topk_index   // [n_query][n_probes][k]
) {
    __shared__ float s_query_norm[QueriesPerBlock];
    
    const int probe_id = blockIdx.x;
    if (probe_id >= n_total_clusters) return;
    
    const int vector_offset = d_probe_vector_offset[probe_id];
    const int vector_count = d_probe_vector_count[probe_id];
    
    const int probe_query_start = d_probe_query_offsets[probe_id];
    const int probe_query_end = d_probe_query_offsets[probe_id + 1];
    const int probe_n_queries = probe_query_end - probe_query_start;
    
    const int query_batch_id = blockIdx.y;
    const int batch_query_start = query_batch_id * QueriesPerBlock;
    const int batch_query_end = min(batch_query_start + QueriesPerBlock, probe_n_queries);
    
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = laneId();
    const int local_query_idx = batch_query_start + warp_id;
    
    if (local_query_idx >= batch_query_end) return;
    
    const int probe_query_idx = probe_query_start + local_query_idx;
    const int query_global_id = d_probe_queries[probe_query_idx];
    const int probe_index_in_query = d_probe_query_probe_indices[probe_query_idx];
    
    // 边界检查：probe_index_in_query 应该在 [0, n_probes) 范围内
    if (probe_index_in_query < 0 || probe_index_in_query >= n_probes) {
        return;  // 无效的probe索引
    }
    
    if (warp_id < QueriesPerBlock && lane == 0) {
        s_query_norm[warp_id] = d_query_norm[query_global_id];
    }
    __syncthreads();
    
    // 检查 query norm 是否有效（所有线程都需要检查）
    if (warp_id < QueriesPerBlock && s_query_norm[warp_id] < 1e-6f) {
        // 如果 norm 无效，输出 dummy 值
        // 输出位置：[n_query][n_probes][k]
        float* row_dist = d_topk_dist + (query_global_id * n_probes + probe_index_in_query) * k;
        int* row_idx = d_topk_index + (query_global_id * n_probes + probe_index_in_query) * k;
        if (lane == 0) {
            for (int i = 0; i < k; ++i) {
                row_dist[i] = FLT_MAX;
                row_idx[i] = -1;
            }
        }
        return;
    }
    
    const float* query_global_ptr = d_query_group + query_global_id * n_dim;
    const bool query_ptr_aligned = (reinterpret_cast<uintptr_t>(query_global_ptr) & (sizeof(float4) - 1)) == 0;
    const bool data_ptr_aligned = (reinterpret_cast<uintptr_t>(d_cluster_vector) & (sizeof(float4) - 1)) == 0;
    const bool prefer_vec4 = query_ptr_aligned && data_ptr_aligned && ((n_dim & 3) == 0);
    
    using WarpSortBase = pgvector::warpsort::WarpSort<Capacity, Ascending, float, int>;
    const float dummy_val = WarpSortBase::kDummy();
    
    WarpSortFiltered<Capacity, Ascending, float, int> queue(k);
    
    int max_iterations = (vector_count + kWarpSize - 1) / kWarpSize;
    
    if (prefer_vec4) {
        for (int iter = 0; iter < max_iterations; ++iter) {
            int vec_idx = vector_offset + iter * kWarpSize + lane;
            bool has_valid_vec = (vec_idx < vector_offset + vector_count);
            
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
            int vec_idx = vector_offset + iter * kWarpSize + lane;
            bool has_valid_vec = (vec_idx < vector_offset + vector_count);
            
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
    
    // 输出位置：[n_query][n_probes][k]
    // 每个probe的结果写入独立位置，避免冲突
    float* row_dist = d_topk_dist + (query_global_id * n_probes + probe_index_in_query) * k;
    int* row_idx = d_topk_index + (query_global_id * n_probes + probe_index_in_query) * k;
    queue.store(row_dist, row_idx);
}

/**
 * Launch 函数：固定 probe 版本
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_indexed_inner_product_with_topk_kernel_v3_fixed_probe(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_probe_queries,
    int* __restrict__ d_probe_query_offsets,
    int* __restrict__ d_probe_query_probe_indices,
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_total_clusters,  // 总的cluster数量（用于grid.x和检查probe_id）
    int n_probes,  // 每个query的probe数量（用于检查probe_index_in_query和计算输出位置）
    int max_queries_per_probe,
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream) {
    
    // 计算每个 probe 的最大 query batch 数
    // 直接从 max_queries_per_probe 计算，避免 host-device 数据复制
    int max_query_batches = (max_queries_per_probe + QueriesPerBlock - 1) / QueriesPerBlock;
    
    dim3 grid(n_total_clusters, max_query_batches, 1);
    
    // 统一使用 generic 版本，支持任意维度
    indexed_inner_product_with_topk_kernel_v3_fixed_probe_generic<Capacity, Ascending, QueriesPerBlock>
        <<<grid, block, 0, stream>>>(
        d_query_group,
        d_cluster_vector,
        d_probe_vector_offset,
        d_probe_vector_count,
        d_probe_queries,
        d_probe_query_offsets,
        d_probe_query_probe_indices,
        d_query_norm,
        d_cluster_vector_norm,
        n_total_clusters,
        n_probes,
        max_queries_per_probe,
        n_dim,
        k,
        d_topk_dist,
        d_topk_index
    );
}

// 显式实例化常用的模板参数组合
template void launch_indexed_inner_product_with_topk_kernel_v3_fixed_probe<64, true, 8>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, float*, float*, int, int, int, int, float*, int*, cudaStream_t);

template void launch_indexed_inner_product_with_topk_kernel_v3_fixed_probe<128, true, 8>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, float*, float*, int, int, int, int, float*, int*, cudaStream_t);

template void launch_indexed_inner_product_with_topk_kernel_v3_fixed_probe<256, true, 8>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, float*, float*, int, int, int, int, float*, int*, cudaStream_t);

