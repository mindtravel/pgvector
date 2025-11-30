#include "indexed_gemm.cuh"
#include "inner_product_utils.cuh"
#include "../pch.h"
#include "../warpsortfilter/warpsort_utils.cuh"
#include "../warpsortfilter/warpsort.cuh"
#include <cfloat>
#include <stdint.h>
#include <vector>
#include <algorithm>

using namespace pgvector::warpsort_utils;
using namespace pgvector::warpsort;

/**
 * v5版本：Entry-based线程模型
 * 每个entry包含一个cluster和一组query（8个或4个）
 * grid维度就是n_entry，每个block处理一个entry
 */

/**
 * Entry结构：在GPU上存储entry信息
 * d_entry_cluster_id: 每个entry对应的cluster_id [n_entry]
 * d_entry_query_start: 每个entry的query起始位置 [n_entry]
 * d_entry_query_count: 每个entry的query数量 [n_entry]
 * d_entry_queries: 所有entry的query列表（扁平化）[total_queries_in_entries]
 * d_entry_probe_indices: 每个entry-query对中probe在query中的索引 [total_queries_in_entries]
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
__global__ __launch_bounds__(256, 1) 
void indexed_inner_product_with_topk_kernel_v5_entry_based(
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_entry_cluster_id,  // [n_entry]
    int* __restrict__ d_entry_query_start,  // [n_entry]
    int* __restrict__ d_entry_query_count,  // [n_entry]
    int* __restrict__ d_entry_queries,  // [total_queries_in_entries]
    int* __restrict__ d_entry_probe_indices,  // [total_queries_in_entries]
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_entry,  // entry总数
    int n_probes,  // 每个query的probe数量（用于检查probe_index_in_query和计算输出位置）
    int n_dim,
    int k,
    float* __restrict__ d_topk_dist,  // [n_query][n_probes][k]
    int* __restrict__ d_topk_index   // [n_query][n_probes][k]
) {
    __shared__ float s_query_norm[QueriesPerBlock];
    
    const int entry_id = blockIdx.x;
    if (entry_id >= n_entry) return;
    
    const int cluster_id = d_entry_cluster_id[entry_id];
    const int vector_offset = d_probe_vector_offset[cluster_id];
    const int vector_count = d_probe_vector_count[cluster_id];
    
    const int entry_query_start = d_entry_query_start[entry_id];
    const int entry_query_count = d_entry_query_count[entry_id];
    // const int entry_query_end = entry_query_start + entry_query_count;
    
    const int local_query_idx = threadIdx.x / kWarpSize;
    const int lane = laneId();
    
    if (local_query_idx >= entry_query_count) return;
    
    const int entry_query_idx = entry_query_start + local_query_idx;
    
    // 边界检查：确保 entry_query_idx 在有效范围内
    // 注意：这里假设 d_entry_queries 和 d_entry_probe_indices 的大小至少为 total_entries
    // 而 entry_query_idx 的范围是 [entry_query_start, entry_query_start + entry_query_count)
    const int query_global_id = d_entry_queries[entry_query_idx];
    const int probe_index_in_query = d_entry_probe_indices[entry_query_idx];
    
    // 边界检查：确保 query_global_id 在有效范围内
    // 注意：这里假设 n_query 是已知的，但kernel参数中没有传递
    // 暂时依赖调用者确保 query_global_id 的有效性
    
    if (lane == 0) {
        s_query_norm[local_query_idx] = d_query_norm[query_global_id];
    }
    __syncthreads();
    
    // 检查 query norm 是否有效（所有线程都需要检查）
    if (s_query_norm[local_query_idx] < 1e-6f) return;
    
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
                    float cos_similarity = dot_product / (s_query_norm[local_query_idx] * data_norm);
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
                    float cos_similarity = dot_product / (s_query_norm[local_query_idx] * data_norm);
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
 * Launch 函数：v5 entry-based版本
 */
template<int Capacity, bool Ascending, int QueriesPerBlock>
void launch_indexed_inner_product_with_topk_kernel_v5_entry_based(
    dim3 block,
    int n_dim,
    float* __restrict__ d_query_group,
    float* __restrict__ d_cluster_vector,
    int* __restrict__ d_probe_vector_offset,
    int* __restrict__ d_probe_vector_count,
    int* __restrict__ d_entry_cluster_id,  // [n_entry]
    int* __restrict__ d_entry_query_start,  // [n_entry]
    int* __restrict__ d_entry_query_count,  // [n_entry]
    int* __restrict__ d_entry_queries,  // [total_queries_in_entries]
    int* __restrict__ d_entry_probe_indices,  // [total_queries_in_entries]
    float* __restrict__ d_query_norm,
    float* __restrict__ d_cluster_vector_norm,
    int n_entry,  // entry总数
    int n_probes,  // 每个query的probe数量
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index,
    cudaStream_t stream) {
    
    dim3 grid(n_entry, 1, 1);
    
    // 统一使用 generic 版本，支持任意维度
    indexed_inner_product_with_topk_kernel_v5_entry_based<Capacity, Ascending, QueriesPerBlock>
        <<<grid, block, 0, stream>>>(
        d_query_group,
        d_cluster_vector,
        d_probe_vector_offset,
        d_probe_vector_count,
        d_entry_cluster_id,
        d_entry_query_start,
        d_entry_query_count,
        d_entry_queries,
        d_entry_probe_indices,
        d_query_norm,
        d_cluster_vector_norm,
        n_entry,
        n_probes,
        n_dim,
        k,
        d_topk_dist,
        d_topk_index
    );
}

// 显式实例化常用的模板参数组合
// QueriesPerBlock=1
template void launch_indexed_inner_product_with_topk_kernel_v5_entry_based<64, true, 1>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, int*, int*, float*, float*, int, int, int, float*, int*, cudaStream_t);

template void launch_indexed_inner_product_with_topk_kernel_v5_entry_based<128, true, 1>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, int*, int*, float*, float*, int, int, int, float*, int*, cudaStream_t);

template void launch_indexed_inner_product_with_topk_kernel_v5_entry_based<256, true, 1>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, int*, int*, float*, float*, int, int, int, float*, int*, cudaStream_t);

// QueriesPerBlock=8
template void launch_indexed_inner_product_with_topk_kernel_v5_entry_based<64, true, 8>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, int*, int*, float*, float*, int, int, int, float*, int*, cudaStream_t);

template void launch_indexed_inner_product_with_topk_kernel_v5_entry_based<128, true, 8>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, int*, int*, float*, float*, int, int, int, float*, int*, cudaStream_t);

template void launch_indexed_inner_product_with_topk_kernel_v5_entry_based<256, true, 8>(
    dim3, int, float*, float*, int*, int*, int*, int*, int*, int*, int*, float*, float*, int, int, int, float*, int*, cudaStream_t);

