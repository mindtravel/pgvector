#include "indexed_gemm.cuh"
#include "../pch.h"
#include "../warpsortfilter/warpsort_utils.cuh"
#include "../warpsortfilter/warpsort.cuh"
#include "../l2norm/l2norm.cuh"

using namespace pgvector::warpsort_utils;
using namespace pgvector::warpsort;

/**
 * 流式内积计算 + top-k选择kernel（v1版本，基于cuVS线程模型）
 * 
 * 线程模型（借鉴cuVS）:
 * - 每个block处理一个query (blockIdx.y = query_id)
 * - 每个block处理该query对应的多个cluster (blockIdx.x循环遍历)
 * - block内的warps分摊计算该query与cluster中向量的距离
 * - 使用WarpSortFiltered流式维护topk，最后合并所有warp的结果
 * 
 * 关键优化：
 * - 在kernel内部维护warp-sort queue，流式加入候选
 * - 直接写入最终输出缓冲区 [n_query, k]，无需中间缓冲区
 * - 内存占用从 O(n_query * max_candidates) 降至 O(n_query * k)
 * - 使用共享内存缓存query向量，减少全局内存访问
 * 
 * @tparam Capacity warp-sort queue的容量（必须是2的幂，且 > k）
 */
template<int Capacity, bool Ascending>
__global__ void indexed_inner_product_with_topk_kernel(
    const float* __restrict__ d_query_group,
    const float* __restrict__ d_cluster_vector,
    const int* __restrict__ d_cluster_query_offset,
    const int* __restrict__ d_cluster_query_data,
    const int* __restrict__ d_cluster_vector_index,  // 每个cluster在全局向量数组中的连续起始位置
    const int* __restrict__ d_cluster_vector_num,
    const float* __restrict__ d_query_norm,
    const float* __restrict__ d_cluster_vector_norm,
    int n_query,
    int distinct_cluster_count,
    int n_dim,
    int tol_vector,
    int k,
    float* __restrict__ d_topk_dist,
    int* __restrict__ d_topk_index
) {
    // 每个block处理一个query
    const int query_id = blockIdx.y;
    if (query_id >= n_query) return;
    
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
    // 每个warp需要存储 k 个距离和索引
    float* warp_dist_shared = reinterpret_cast<float*>(shared_mem + query_smem_elems * sizeof(float));
    int* warp_idx_shared = reinterpret_cast<int*>(warp_dist_shared + n_warps * k);
    
    using WarpSortBase = pgvector::warpsort::WarpSort<Capacity, Ascending, float, int>;
    const float dummy_val = WarpSortBase::kDummy();
    
    // 每个warp维护一个全局的局部topk queue（用于累积所有处理过的cluster的结果）
    WarpSortFiltered<Capacity, Ascending, float, int> queue(k);
    
    // 遍历所有cluster，每个warp处理一部分cluster
    // 注意：每个warp需要确保处理完所有分配给它的cluster，即使某些cluster不包含当前query
    // bool warp_has_valid_data = false;  // 标记该warp是否处理过包含当前query的cluster
    int warp_processed_cluster_count = 0;  // 该warp处理过的包含该query的cluster数量（用于调试）
    
    // 调试输出：打印所有cluster的query范围（只在第一个block的第一个warp的第一个线程打印）
    // if (warp_id == 0 && lane == 0 && query_id == 0) {
    //     printf("[DEBUG] All clusters info for verification: ");
    //     for (int c = 0; c < distinct_cluster_count; c++) {
    //         int qs = d_cluster_query_offset[c];
    //         int qe = d_cluster_query_offset[c + 1];
    //         printf("C%d[%d:%d]={", c, qs, qe);
    //         for (int q = qs; q < qe; q++) {
    //             printf("%d,", d_cluster_query_data[q]);
    //         }
    //         printf("} ");
    //     }
    //     printf("\n");
    // }
    __syncthreads();
    
    for (int cluster_idx = warp_id; cluster_idx < distinct_cluster_count; cluster_idx += n_warps) {
        // 调试：显示每个warp检查哪些cluster
        // if (lane == 0 && warp_id < 2) {
        //     printf("[DEBUG] Query %d, Warp %d: Checking cluster %d\n", query_id, warp_id, cluster_idx);
        // }
        __syncwarp();
        
        // 检查该cluster是否包含当前query
        int query_start = d_cluster_query_offset[cluster_idx];
        int query_end = d_cluster_query_offset[cluster_idx + 1];
        
        // 边界检查：如果该cluster没有关联任何query，直接跳过
        if (query_start >= query_end) {
            // if (warp_id < 2 && lane == 0) {
            //     printf("[DEBUG] Query %d, Warp %d: Cluster %d has no queries (start=%d, end=%d), skipping\n",
            //            query_id, warp_id, cluster_idx, query_start, query_end);
            // }
            continue;
        }
        
        // 使用warp内的线程并行查找（提高效率）
        bool query_in_cluster = false;
        for (int q = query_start + lane; q < query_end; q += kWarpSize) {
            if (q < query_end && d_cluster_query_data[q] == query_id) {
                query_in_cluster = true;
            }
        }
        // 使用warp内的ballot操作合并结果
        query_in_cluster = (__ballot_sync(0xffffffff, query_in_cluster) != 0);
        
        if (!query_in_cluster) {
            // 该cluster不包含当前query，继续下一个cluster
            // 注意：这里不要往queue里添加任何数据，因为该cluster与当前query无关
            // if (warp_id < 2 && lane == 0) {  // 只打印前2个warp的信息，减少输出
            //     printf("[DEBUG] Query %d, Warp %d: Cluster %d does NOT contain query %d (queries in cluster: ", 
            //            query_id, warp_id, cluster_idx, query_id);
            //     for (int q = query_start; q < query_end; q++) {
            //         printf("%d,", d_cluster_query_data[q]);
            //     }
            //     printf("), skipping\n");
            // }
            continue;
        }
        
        // 标记该warp处理过有效数据
        // warp_has_valid_data = true;
        warp_processed_cluster_count++;
        
        // if (warp_id < 2 && lane == 0) {
        //     printf("[DEBUG] Query %d, Warp %d: Processing cluster %d (this is the %d-th cluster for this warp)\n",
        //            query_id, warp_id, cluster_idx, warp_processed_cluster_count);
        // }
        
        // 获取该cluster的向量信息（连续存储）
        int vector_start_idx = d_cluster_vector_index[cluster_idx];
        int vector_count = d_cluster_vector_num[cluster_idx];
        
        // 边界检查
        if (vector_start_idx < 0 || vector_count <= 0 || 
            vector_start_idx + vector_count > tol_vector) {
            // if (warp_id < 2 && lane == 0) {
            //     printf("[DEBUG] Query %d, Warp %d: Cluster %d has invalid vector info (start=%d, count=%d, tol=%d), skipping\n",
            //            query_id, warp_id, cluster_idx, vector_start_idx, vector_count, tol_vector);
            // }
            continue;
        }
        
        // 调试：打印cluster的向量信息
        // if (warp_id < 2 && lane == 0) {
        //     printf("[DEBUG] Query %d, Warp %d: Cluster %d has %d vectors, starting at idx=%d\n",
        //            query_id, warp_id, cluster_idx, vector_count, vector_start_idx);
        // }
        
        // 该warp处理该cluster中的所有向量
        // 计算需要处理的向量数量（每个线程处理一部分）
        int max_iterations = (vector_count + kWarpSize - 1) / kWarpSize;
        
        // 调试：打印迭代信息
        // if (warp_id < 2 && lane == 0) {
        //     printf("[DEBUG] Query %d, Warp %d: Cluster %d processing: vector_count=%d, max_iterations=%d\n",
        //            query_id, warp_id, cluster_idx, vector_count, max_iterations);
        // }
        
        // 统计处理了多少向量（用于调试）
        int processed_vec_count = 0;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            int vec_idx = iter * kWarpSize + lane;
            bool has_valid_vec = (vec_idx < vector_count);
            
            if (!has_valid_vec) {
                // 无效向量索引，添加dummy值但所有线程都必须调用add()
                queue.add(dummy_val, -1);
            } else {
                int global_vec_idx = vector_start_idx + vec_idx;
                
                // 边界检查
                if (global_vec_idx < 0 || global_vec_idx >= tol_vector) {
                    queue.add(dummy_val, -1);
                } else {
                    // 计算内积
                    float dot_product = 0.0f;
                    
                    if (use_shmem) {
                        // 使用共享内存中的query
                        #pragma unroll 4
                        for (int dim = 0; dim < n_dim; dim++) {
                            dot_product += query_shared[dim] * 
                                          d_cluster_vector[global_vec_idx * n_dim + dim];
                        }
                    } else {
                        // 使用共享内存中的部分query + 全局内存中的剩余部分
                        #pragma unroll 4
                        for (int dim = 0; dim < query_smem_elems; dim++) {
                            dot_product += query_shared[dim] * 
                                          d_cluster_vector[global_vec_idx * n_dim + dim];
                        }
                        #pragma unroll 4
                        for (int dim = query_smem_elems; dim < n_dim; dim++) {
                            dot_product += d_query_group[query_id * n_dim + dim] * 
                                          d_cluster_vector[global_vec_idx * n_dim + dim];
                        }
                    }
                    
                    // 获取向量norm
                    float data_norm = d_cluster_vector_norm[global_vec_idx];
                    if (data_norm < 1e-6f) {
                        queue.add(dummy_val, -1);
                    } else {
                        // 计算余弦距离
                        float cos_similarity = dot_product / (query_norm * data_norm);
                        float cos_distance = 1.0f - cos_similarity;
                        
                        // 调试输出（打印所有处理的向量，但要控制输出量）
                        // if (warp_id < 2 && lane < vector_count && global_vec_idx < 20) {  // 打印前20个向量
                        //     printf("[DEBUG] Query %d: Cluster %d, Vec %d (global_idx=%d, lane=%d, iter=%d): dot=%.5f, cos_sim=%.5f, cos_dist=%.5f -> queue.add()\n",
                        //            query_id, cluster_idx, vec_idx, global_vec_idx, lane, iter, dot_product, cos_similarity, cos_distance);
                        // }
                        
                        processed_vec_count++;
                        queue.add(cos_distance, global_vec_idx);
                    }
                }
            }
        }
        
        // 调试：统计warp处理了多少向量（使用warp内的reduction）
        __syncwarp();
        // 每个线程的processed_vec_count累加（使用warp内的shuffle reduction）
        int total_processed = processed_vec_count;
        for (int offset = 16; offset > 0; offset /= 2) {
            int other = __shfl_sync(0xffffffff, processed_vec_count, lane + offset, 32);
            if ((lane + offset) < 32 && (lane + offset) < vector_count) {
                total_processed += other;
            }
        }
        // 只让lane 0打印（但total_processed可能不准确，因为shuffle reduction的逻辑不对）
        // 简化：使用ballot统计有多少线程处理了有效向量
        unsigned int processed_mask = __ballot_sync(0xffffffff, processed_vec_count > 0);
        int threads_with_vectors = __popc(processed_mask);
        
        // if (warp_id < 2 && lane == 0) {
        //     printf("[DEBUG] Query %d, Warp %d: Cluster %d - %d threads processed vectors, total_vec_count=%d, max_iterations=%d\n",
        //            query_id, warp_id, cluster_idx, threads_with_vectors, vector_count, max_iterations);
        // }
        
        // 完成该cluster的处理（但queue继续累积，用于下一个cluster）
        __syncwarp();
    }
    
    // 所有cluster处理完后，完成queue处理
    // 注意：如果该warp没有处理任何包含当前query的cluster，queue仍然是初始状态（全是dummy值）
    queue.done();
    
    // 所有warp完成各自cluster的处理后，将局部topk写入共享内存
    __syncwarp();
    float* warp_dist = warp_dist_shared + warp_id * k;
    int* warp_idx = warp_idx_shared + warp_id * k;
    
    // 注意：store 需要所有线程参与（虽然只有部分线程实际存储数据）
    // 如果warp_has_valid_data为false，store的结果都是dummy值（INFINITY），在合并时会被跳过
    
    // 在store之前，先打印每个线程存储的位置（用于调试）
    __syncwarp();
    // if (warp_id == 0 && lane < k) {
    //     // 计算store时使用的索引（与WarpSort::store方法一致）
    //     using WarpSortBase = pgvector::warpsort::WarpSort<Capacity, Ascending, float, int>;
    //     constexpr int kWarpWidth = (Capacity < 32) ? Capacity : 32;
    //     int idx = (lane % kWarpWidth);
    //     printf("[DEBUG] Query %d, Warp %d: Before store, lane %d will store to idx=%d (kWarpWidth=%d)\n",
    //            query_id, warp_id, lane, idx, kWarpWidth);
    // }
    __syncwarp();
    
    queue.store(warp_dist, warp_idx);
    
    // 同步后验证store的结果
    __syncwarp();
    
    // 调试输出：打印每个warp的局部topk结果
    // if (warp_id == 0 && lane == 0) {
    //     printf("[DEBUG] Query %d, Warp %d: Processed %d clusters, has_valid_data=%d\n",
    //            query_id, warp_id, warp_processed_cluster_count, warp_has_valid_data);
    //     printf("[DEBUG] Query %d, Warp %d local topk AFTER store: ", query_id, warp_id);
    //     for (int i = 0; i < k; i++) {
    //         if (warp_dist[i] != INFINITY && warp_dist[i] >= 0.0f && warp_dist[i] <= 2.0f) {
    //             printf("[%d] dist=%.5f idx=%d, ", i, warp_dist[i], warp_idx[i]);
    //         } else {
    //             printf("[%d] invalid(%.5f), ", i, warp_dist[i]);
    //         }
    //     }
    //     printf("\n");
    // }
    
    // 同步后打印所有warp的结果
    __syncthreads();
    // if (warp_id < n_warps && lane == 0) {
    //     float* w_dist = warp_dist_shared + warp_id * k;
    //     int* w_idx = warp_idx_shared + warp_id * k;
    //     printf("[DEBUG] Query %d, Warp %d stored topk: ", query_id, warp_id);
    //     for (int i = 0; i < k; i++) {
    //         if (w_dist[i] != INFINITY && w_dist[i] >= 0.0f && w_dist[i] <= 2.0f) {
    //             printf("[%d] dist=%.5f idx=%d, ", i, w_dist[i], w_idx[i]);
    //         } else {
    //             printf("[%d] invalid(%.5f), ", i, w_dist[i]);
    //         }
    //     }
    //     printf("\n");
    // }
    __syncthreads();
    
    // 同步所有warp
    __syncthreads();
    
    // 合并所有warp的局部topk（由第一个warp执行合并，需要所有线程参与）
    if (warp_id == 0) {
        // 使用第一个warp合并所有warp的局部topk
        WarpSortFiltered<Capacity, Ascending, float, int> final_queue(k);
        
        // 收集所有warp的局部topk
        // 确保所有线程都执行相同次数的add()调用
        // 对于每个warp的每个元素，所有线程都需要参与add()
        for (int w = 0; w < n_warps; w++) {
            float* w_dist = warp_dist_shared + w * k;
            int* w_idx = warp_idx_shared + w * k;
            
            // 每个线程处理一部分元素，确保所有线程都调用add()
            // 对于 k=2，max_iterations=1，只有 lane 0-1 会处理有效元素，lane 2-31 处理无效元素
            int max_iterations = (k + kWarpSize - 1) / kWarpSize;
            
            // 在合并每个warp之前，先打印该warp的所有数据（由lane 0统一打印）
            // if (lane == 0) {
            //     printf("[DEBUG] Query %d: About to merge warp %d's topk: ", query_id, w);
            //     for (int j = 0; j < k; j++) {
            //         if (w_dist[j] != INFINITY && w_dist[j] >= 0.0f && w_dist[j] <= 2.0f) {
            //             printf("[%d] dist=%.5f idx=%d, ", j, w_dist[j], w_idx[j]);
            //         } else {
            //             printf("[%d] invalid(%.5f), ", j, w_dist[j]);
            //         }
            //     }
            //     printf("\n");
            // }
            __syncwarp();
            
            for (int iter = 0; iter < max_iterations; iter++) {
                int i = iter * kWarpSize + lane;
                bool is_valid = (i < k);
                
                // 所有线程都必须调用add()，即使值无效
                // 检查是否为有效元素：
                // 1. 索引有效 (is_valid)
                // 2. 不是dummy值（对于Ascending=true，dummy值是INFINITY）
                // 3. 距离值应该是合理的（在0到2之间，因为余弦距离=1-cos_similarity）
                if (is_valid && w_dist[i] != INFINITY && w_dist[i] >= 0.0f && w_dist[i] <= 2.0f) {
                    // 有效元素，加入final_queue
                    // 调试输出：所有线程都打印（但会被同步，可能输出顺序乱）
                    // printf("[DEBUG] Query %d merging: warp %d, lane %d, elem %d: dist=%.5f idx=%d (VALID, adding)\n",
                    //        query_id, w, lane, i, w_dist[i], w_idx[i]);
                    final_queue.add(w_dist[i], w_idx[i]);
                } else {
                    // 无效元素或dummy值，但仍需调用add()保持同步
                    // if (is_valid) {
                    //     printf("[DEBUG] Query %d merging: warp %d, lane %d, elem %d: dist=%.5f idx=%d (INVALID, skipping but adding dummy)\n",
                    //            query_id, w, lane, i, w_dist[i], w_idx[i]);
                    // }
                    final_queue.add(INFINITY, -1);
                }
            }
        }
        
        // 完成合并
        __syncwarp();
        
        // 在 done() 之前打印统计信息
        // if (lane == 0) {
        //     printf("[DEBUG] Query %d: Before final_queue.done(), total elements added to final_queue:\n", query_id);
        //     printf("  - From warp 0: 2 elements (0.70242, 1.04023 for Q0; 0.37890, 0.56416 for Q1)\n");
        //     printf("  - From warp 1: 2 elements (0.33034, 0.47693 for Q0; 0.76149, 1.42689 for Q1)\n");
        //     printf("  - Expected top-2 for Q0: 0.33034, 0.47693 (correct!)\n");
        //     printf("  - Expected top-2 for Q1: 0.37890, 0.56416 (correct!)\n");
        // }
        
        final_queue.done();
        
        // 在存储之前，先检查 final_queue 的内容（通过共享内存）
        // 注意：我们不能直接访问 final_queue 的内部数组，但可以在存储后验证
        __syncwarp();
        
        // 写入最终结果
        float* row_dist = d_topk_dist + query_id * k;
        int* row_idx = d_topk_index + query_id * k;
        final_queue.store(row_dist, row_idx);
        
        // 调试输出：打印最终结果（所有线程都打印，但会被同步）
        __syncwarp();
        // if (lane < k) {
        //     printf("[DEBUG] Query %d FINAL topk[%d]: dist=%.5f idx=%d (from lane %d)\n",
        //            query_id, lane, row_dist[lane], row_idx[lane], lane);
        // }
        __syncwarp();
        
        // 由 lane 0 统一打印最终结果（格式化的）
        // if (lane == 0) {
        //     printf("[DEBUG] Query %d FINAL topk (formatted): ", query_id);
        //     for (int i = 0; i < k; i++) {
        //         printf("[%d] dist=%.5f idx=%d, ", i, row_dist[i], row_idx[i]);
        //     }
        //     printf("\n");
        //     
        //     // 验证最终结果是否正确（使用浮点数容差）
        //     const float EPS = 1e-5f;
        //     bool result_correct = true;
        //     bool sorted_correct = true;
        //     
        //     // 检查排序是否正确（距离应该递增）
        //     for (int i = 1; i < k; i++) {
        //         if (row_dist[i] < row_dist[i-1] - EPS) {
        //             sorted_correct = false;
        //             printf("[DEBUG] Query %d: Sorting WRONG! dist[%d]=%.5f < dist[%d]=%.5f\n",
        //                    query_id, i, row_dist[i], i-1, row_dist[i-1]);
        //         }
        //     }
        //     
        //     if (sorted_correct) {
        //         printf("[DEBUG] Query %d: Final result is SORTED correctly: [0]=%.5f idx=%d, [1]=%.5f idx=%d\n",
        //                query_id, row_dist[0], row_idx[0], row_dist[1], row_idx[1]);
        //     } else {
        //         printf("[DEBUG] Query %d: Final result sorting is WRONG!\n", query_id);
        //     }
        //     
        //     // 注意：这里不验证具体值，因为可能与CPU实现有细微差异
        //     // 主要验证是否从所有候选元素中选择了最小的k个
        //     printf("[DEBUG] Query %d: Verification complete. GPU computed top-%d from all candidates.\n",
        //            query_id, k);
        // }
    }
}

// 显式实例化（已禁用，只使用v3_fixed_probe版本）
/*
template __global__ void indexed_inner_product_with_topk_kernel<64, true>(
    const float* __restrict__, const float* __restrict__, const int* __restrict__, const int* __restrict__,
    const int* __restrict__, const int* __restrict__, const float* __restrict__, const float* __restrict__,
    int, int, int, int, int,
    float* __restrict__, int* __restrict__);

template __global__ void indexed_inner_product_with_topk_kernel<128, true>(
    const float* __restrict__, const float* __restrict__, const int* __restrict__, const int* __restrict__,
    const int* __restrict__, const int* __restrict__, const float* __restrict__, const float* __restrict__,
    int, int, int, int, int,
    float* __restrict__, int* __restrict__);

template __global__ void indexed_inner_product_with_topk_kernel<256, true>(
    const float* __restrict__, const float* __restrict__, const int* __restrict__, const int* __restrict__,
    const int* __restrict__, const int* __restrict__, const float* __restrict__, const float* __restrict__,
    int, int, int, int, int,
    float* __restrict__, int* __restrict__);
*/
