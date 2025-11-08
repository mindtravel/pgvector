#include "indexed_gemm.cuh"
#include "../pch.h"
#include "../warpsortfilter/warpsort_utils.cuh"
#include "../warpsortfilter/warpsort.cuh"

using namespace pgvector::warpsort_utils;
using namespace pgvector::warpsort;

/**
 * 流式内积计算 + top-k选择kernel（v2版本：优化数据上传）
 * 
 * 线程模型（借鉴cuVS）:
 * - 每个block处理一个query (blockIdx.y = query_id)
 * - 根据query_cluster_offset找到该query涉及的所有cluster
 * - block内的warps分摊计算该query与cluster中向量的距离
 * - 使用WarpSortFiltered流式维护topk，最后合并所有warp的结果
 * 
 * 关键优化：
 * - cluster向量连续存储
 * - 预计算l2norm（避免重复计算）
 * 
 * @tparam Capacity warp-sort queue的容量（必须是2的幂，且 > k）
 */
template<int Capacity, bool Ascending>
__global__ void indexed_inner_product_with_topk_kernel_v2(
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
    // 每个block处理一个query
    const int query_id = d_query_index[blockIdx.y];
    if (blockIdx.y >= n_selected_querys) return;
    
    const int thread_idx = threadIdx.x;
    const int warp_id = thread_idx / kWarpSize;
    const int lane = laneId();
    const int n_warps = blockDim.x / kWarpSize;
    
    // 使用共享内存缓存query向量（最大128个元素）
    extern __shared__ char shared_mem[];
    float* query_shared = reinterpret_cast<float*>(shared_mem);
    const int query_smem_elems = (n_dim <= 128) ? n_dim : 128;
    const bool use_shmem = (n_dim <= 128);
    
    // 加载query到共享内存（所有线程协作加载）
    for (int i = thread_idx; i < query_smem_elems; i += blockDim.x) {
        if (i < n_dim) {
            query_shared[i] = d_query_group[query_id * n_dim + i];
        }
    }
    __syncthreads();
    
    // 获取query的norm
    const float query_norm = d_query_norm[query_id];
    if (query_norm < 1e-6f) return;  // 跳过无效query
    
    // 使用共享内存存储每个warp的局部topk结果（用于最终合并）
    float* warp_dist_shared = reinterpret_cast<float*>(shared_mem + query_smem_elems * sizeof(float));
    int* warp_idx_shared = reinterpret_cast<int*>(warp_dist_shared + n_warps * k);
    
    using WarpSortBase = pgvector::warpsort::WarpSort<Capacity, Ascending, float, int>;
    const float dummy_val = WarpSortBase::kDummy();
    
    // 每个warp维护一个全局的局部topk queue（用于累积所有处理过的cluster的结果）
    WarpSortFiltered<Capacity, Ascending, float, int> queue(k);
       
    __syncwarp();
        
    // 该warp处理该cluster中的所有向量
    int max_iterations = (n_selected_vectors + kWarpSize - 1) / kWarpSize;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        int vec_idx = iter * kWarpSize + lane;
        bool has_valid_vec = (vec_idx < n_selected_vectors);
        
        if (!has_valid_vec) {
            // 无效向量索引，添加dummy值但所有线程都必须调用add()
            queue.add(dummy_val, -1);
        } else {     
            // 计算内积
            float dot_product = 0.0f;
            
            if (use_shmem) {
                // 使用共享内存中的query
                #pragma unroll 4
                for (int dim = 0; dim < n_dim; dim++) {
                    dot_product += query_shared[dim] * 
                                    d_cluster_vector[vec_idx * n_dim + dim];
                }
            } else {
                // 使用共享内存中的部分query + 全局内存中的剩余部分
                #pragma unroll 4
                for (int dim = 0; dim < query_smem_elems; dim++) {
                    dot_product += query_shared[dim] * 
                                    d_cluster_vector[vec_idx * n_dim + dim];
                }
                #pragma unroll 4
                for (int dim = query_smem_elems; dim < n_dim; dim++) {
                    dot_product += d_query_group[query_id * n_dim + dim] * 
                                    d_cluster_vector[vec_idx * n_dim + dim];
                }
            }
            
            // 获取向量norm（使用在d_cluster_vector中的索引）
            float data_norm = d_cluster_vector_norm[vec_idx];
            if (data_norm < 1e-6f) {
                queue.add(dummy_val, -1);
            } else {
                // 计算余弦距离
                float cos_similarity = dot_product / (query_norm * data_norm);
                float cos_distance = 1.0f - cos_similarity;
                
                // 注意：这里存储的索引是global_vec_idx（在d_cluster_vector中的索引）
                // 如果需要原始全局索引，需要在host端进行映射
                queue.add(cos_distance, vec_idx);
            }
        }
    }
    
    // 完成该cluster的处理（但queue继续累积，用于下一个cluster）
    __syncwarp();
    
    // 所有cluster处理完后，完成queue处理
    queue.done();
    
    // 所有warp完成各自cluster的处理后，将局部topk写入共享内存
    __syncwarp();
    float* warp_dist = warp_dist_shared + warp_id * k;
    int* warp_idx = warp_idx_shared + warp_id * k;
    
    queue.store(warp_dist, warp_idx);
    
    // 同步后验证store的结果
    __syncwarp();
    __syncthreads();
    
    // 合并所有warp的局部topk（由第一个warp执行合并，需要所有线程参与）
    if (warp_id == 0) {
        // 使用第一个warp合并所有warp的局部topk
        WarpSortFiltered<Capacity, Ascending, float, int> final_queue(k);
        
        // 收集所有warp的局部topk
        for (int w = 0; w < n_warps; w++) {
            float* w_dist = warp_dist_shared + w * k;
            int* w_idx = warp_idx_shared + w * k;
            
            // 每个线程处理一部分元素，确保所有线程都调用add()
            int max_iterations = (k + kWarpSize - 1) / kWarpSize;
            __syncwarp();
            
            for (int iter = 0; iter < max_iterations; iter++) {
                int i = iter * kWarpSize + lane;
                bool is_valid = (i < k);
                
                // 所有线程都必须调用add()，即使值无效
                if (is_valid && w_dist[i] != INFINITY && w_dist[i] >= 0.0f && w_dist[i] <= 2.0f) {
                    final_queue.add(w_dist[i], w_idx[i]);
                } else {
                    final_queue.add(INFINITY, -1);
                }
            }
        }
        
        // 完成合并
        __syncwarp();
        final_queue.done();
        __syncwarp();
        
        // 写入最终结果
        float* row_dist = d_topk_dist + query_id * k;
        int* row_idx = d_topk_index + query_id * k;
        final_queue.store(row_dist, row_idx);
    }
}

// 显式实例化
template __global__ void indexed_inner_product_with_topk_kernel_v2<64, true>(
    float* __restrict__, float* __restrict__, int* __restrict__, 
    float* __restrict__, float* __restrict__,
    int, int, int, int,
    float* __restrict__, int* __restrict__);

template __global__ void indexed_inner_product_with_topk_kernel_v2<128, true>(
    float* __restrict__, float* __restrict__, int* __restrict__, 
    float* __restrict__, float* __restrict__,
    int, int, int, int,
    float* __restrict__, int* __restrict__);

template __global__ void indexed_inner_product_with_topk_kernel_v2<256, true>(
    float* __restrict__, float* __restrict__, int* __restrict__, 
    float* __restrict__, float* __restrict__,
    int, int, int, int,
    float* __restrict__, int* __restrict__);


