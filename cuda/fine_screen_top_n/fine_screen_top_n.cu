#include "fine_screen_top_n.cuh"
#include "../l2norm/l2norm.cuh"
#include "../unit_tests/common/test_utils.cuh"
#include "../fusion_cos_topk/fusion_cos_topk.cuh"
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <limits.h>
#include <float.h>

#define ENABLE_CUDA_TIMING 1


// // 声明融合余弦距离top-k计算函数
// extern void cuda_cos_topk_warpsort_fine(
//     const float* d_query_group,
//     const float* d_cluster_vector,
//     const int* d_cluster_query_offset,
//     const int* d_cluster_query_data,
//     const int* d_cluster_vector_index,
//     const int* d_cluster_vector_num,
//     const float* d_query_norm,
//     const float* d_cluster_vector_norm,
//     int* d_topk_index,
//     float* d_topk_dist,
//     int n_query,
//     int n_total_clusters,
//     int n_dim,
//     int n_topn,
//     int n_total_vectors,
//     int max_cluster_vector_count
// );

// /**
//  * 预留的warpsort接口，用于在寄存器中维护局部top-k
//  * 参数待定，实现待定
//  */
// __device__ void cluster_warpsort_topk(
//     float* local_distances,    // 当前cluster的距离数组
//     int* local_indices,        // 对应的索引数组
//     int cluster_vector_count,   // 当前cluster的向量数量
//     int k,                     // top-k数量
//     float* output_distances,   // 输出距离
//     int* output_indices        // 输出索引
// ) {
//     // 实现待定
//     // 这里暂时用简单的排序实现
//     for (int i = 0; i < cluster_vector_count - 1; i++) {
//         for (int j = i + 1; j < cluster_vector_count; j++) {
//             if (local_distances[i] > local_distances[j]) {
//                 // 交换距离
//                 float temp_dist = local_distances[i];
//                 local_distances[i] = local_distances[j];
//                 local_distances[j] = temp_dist;
                
//                 // 交换索引
//                 int temp_idx = local_indices[i];
//                 local_indices[i] = local_indices[j];
//                 local_indices[j] = temp_idx;
//             }
//         }
//     }
    
//     // 复制前k个结果
//     for (int i = 0; i < k && i < cluster_vector_count; i++) {
//         output_distances[i] = local_distances[i];
//         output_indices[i] = local_indices[i];
//     }
// }

// /**
//  * 计算cluster中向量与query的L2距离并选择top-k
//  * 每个block处理一个cluster
//  */
// __global__ void cluster_l2_distance_kernel(
//     const float* __restrict__ d_query_group,
//     const float* __restrict__ d_query_norm,
//     const float* __restrict__ d_cluster_vector,
//     const float* __restrict__ d_cluster_vector_norm,
//     const int* __restrict__ d_query_cluster_group,
//     const int* __restrict__ d_cluster_query_offset,
//     const int* __restrict__ d_cluster_query_data,
//     const int* __restrict__ d_cluster_map,
//     const int* __restrict__ d_cluster_vector_index,
//     const int* __restrict__ d_cluster_vector_num,
//     int n_query, int n_cluster, int n_dim, int n_topn,
//     int max_cluster_vector_count, int n_total_clusters, int n_total_vectors,
//     int* __restrict__ d_query_mutex,
//     int* __restrict__ d_topn_index,
//     float* __restrict__ d_topn_dist
// ) {
//     int cluster_idx = blockIdx.x;
//     int thread_idx = threadIdx.x;
//     if (cluster_idx >= n_total_clusters || thread_idx >= d_cluster_vector_num[cluster_idx]) return;
//     if (thread_idx >= blockDim.x) return;
//     // 共享内存：缓存L2范数和cluster向量数据
//     extern __shared__ float shared_mem[];
//     float* s_query_norm = shared_mem;
//     float* s_cluster_norm = s_query_norm + n_query;
    
//     // 只有第一个线程计算query范围，避免越界
//     int query_start, query_count;
//     if (thread_idx == 0) {
//         // 边界检查：确保不越界访问
//         if (cluster_idx >= n_total_clusters) {
//             query_count = 0;
//         } else {
//             query_start = d_cluster_query_offset[cluster_idx];
            
//             // 对于最后一个cluster，使用总数作为结束位置
//             if (cluster_idx + 1 >= n_total_clusters) {
//                 // 最后一个cluster：query_count = 总query数 - query_start
//                 query_count = n_query - query_start;
//             } else {
//                 query_count = d_cluster_query_offset[cluster_idx + 1] - query_start;
//             }
            
//             // 额外的越界检查
//             if (query_start >= n_query || query_start + query_count > n_query || query_count < 0) {
//                 query_count = 0;
//             }
//         }
//     }
//     if (query_count == 0) return;

//     // 获取当前cluster的向量信息
//     int vector_start_idx = d_cluster_vector_index[cluster_idx];
//     int vector_count = d_cluster_vector_num[cluster_idx];
    
//     // 修复：添加边界检查，确保向量索引有效
//     if (vector_start_idx < 0 || vector_count <= 0 || vector_start_idx + vector_count > n_total_vectors) {
//         return;
//     }
//     __syncthreads();
    
    
//     // 加载L2范数到共享内存
//     if (thread_idx < n_query) {
//         s_query_norm[thread_idx] = d_query_norm[thread_idx];
//     }
//     // 修复：加载当前cluster的向量L2范数，添加边界检查
//     if (thread_idx < vector_count && thread_idx < max_cluster_vector_count) {
//         int global_vec_idx = vector_start_idx + thread_idx;
//         if (global_vec_idx < n_total_vectors) {
//             s_cluster_norm[thread_idx] = d_cluster_vector_norm[global_vec_idx];
//         }
//     }
//     __syncthreads();
    
//     // 每个线程处理cluster中的部分向量
//     int vectors_per_thread = (vector_count + blockDim.x - 1) / blockDim.x;
//     int start_vec = thread_idx * vectors_per_thread;
//     int end_vec = min(start_vec + vectors_per_thread, vector_count);
    
//     // 为每个query计算L2距离并维护局部topk
//     for (int q = 0; q < query_count; q++) {
//         int query_idx = query_start + q;
        
        
        
//         // 计算当前query与cluster中向量的L2距离
//         for (int vec_idx = start_vec; vec_idx < end_vec; vec_idx++) {
//             int global_vec_idx = vector_start_idx + vec_idx;
            
//             // 修复：添加边界检查，确保全局向量索引有效
//             if (global_vec_idx < 0 || global_vec_idx >= n_total_vectors) {
//                 continue;
//             }
            
//             // 计算L2距离的平方（使用L2范数优化）    todo 其实这里也可以提前计算出来 后续看哪个性能更好一点吧
//             float dot_product = 0.0f;
//             for (int dim = 0; dim < n_dim; dim++) {
//                 dot_product += d_query_group[query_idx * n_dim + dim] * 
//                               d_cluster_vector[global_vec_idx * n_dim + dim];
//             }
            
//             // L2距离平方 = ||q||^2 + ||v||^2 - 2*q·v
//             float distance_squared = s_query_norm[query_idx] + s_cluster_norm[vec_idx] - 2.0f * dot_product;
            
//             // 取平方根得到实际距离
//             float distance = sqrtf(fmaxf(0.0f, distance_squared));
            
//             // // 插入到当前query的局部topk中
//             // for (int k = 0; k < n_topn; k++) {
//             //     if (distance < query_local_topk_dist[k]) {
//             //         // 向后移动元素
//             //         for (int m = n_topn - 1; m > k; m--) {
//             //             query_local_topk_dist[m] = query_local_topk_dist[m-1];
//             //             query_local_topk_idx[m] = query_local_topk_idx[m-1];
//             //         }
//             //         // 插入新元素
//             //         query_local_topk_dist[k] = distance;
//             //         query_local_topk_idx[k] = global_vec_idx;
//             //         break;
//             //     }
//             // }
//         }
        
//     }
    
//     __syncthreads();
    
//     // 写入显存对应位置 - 使用原子操作加锁
//     // 每个线程处理自己负责的query范围
    
//     int queries_per_thread = (query_count + blockDim.x - 1) / blockDim.x;
//     int start_query = thread_idx * queries_per_thread;
//     int end_query = min(start_query + queries_per_thread, query_count);
    
//     for (int q = start_query; q < end_query; q++) {
//         int query_idx = query_start + q;
//         if (query_idx >= n_query) continue;
//         // 使用原子操作获取锁
//         while (atomicCAS(&d_query_mutex[query_idx], 0, 1) != 0) {
//             // 自旋等待
//         }
        
//         // 合并局部topk到全局topk
//         // 修复：添加边界检查，确保索引不越界
//         for (int k = 0; k < n_topn && k < vector_count; k++) {
//             // 确保全局向量索引在有效范围内
//             if (query_idx * n_topn + k >= n_query * n_topn) continue;
//             d_topn_index[query_idx * n_topn + k] = vector_start_idx + k;
//             // TODO: 这里应该使用实际计算的距离值，而不是临时值
//             // 需要实现真正的top-k选择逻辑来获取正确的距离
//             d_topn_dist[query_idx * n_topn + k] = 0.0f; // 临时值，需要替换为实际距离
            
//         }
        
//         // 释放锁
//         atomicExch(&d_query_mutex[query_idx], 0);
//     }

// }

void fine_screen_top_n(
    float** h_query_group, int* h_cluster_query_offset, int* h_cluster_query_data,
    int* h_cluster_vector_index, int* h_cluster_vector_num, float** h_cluster_vector,
    int n_query, int n_cluster, int n_total_clusters, int n_dim, int n_topn, int n_total_vectors,
    int** h_query_topn_index, float** h_query_topn_dist
) {
    // 优化：计算每个query的实际候选数量（参考cuVS的实现）
    // 这样可以避免分配大量冗余内存
    // 
    // 方法：遍历所有cluster的query列表，累加每个query关联的cluster的向量数量
    // 时间复杂度：O(n_query * n_cluster)，比嵌套循环更高效
    int* h_num_samples = (int*)calloc(n_query, sizeof(int));  // 初始化为0
    
    for (int c = 0; c < n_total_clusters; c++) {
        int query_start = h_cluster_query_offset[c];
        int query_end = h_cluster_query_offset[c + 1];
        int cluster_vector_count = h_cluster_vector_num[c];
        
        // 遍历该cluster关联的所有query，累加候选数
        for (int idx = query_start; idx < query_end; idx++) {
            int query_idx = h_cluster_query_data[idx];
            if (query_idx >= 0 && query_idx < n_query) {
                h_num_samples[query_idx] += cluster_vector_count;
            }
        }
    }
    
    // 找到最大候选数
    int max_candidates_per_query = 0;
    for (int q = 0; q < n_query; q++) {
        if (h_num_samples[q] > max_candidates_per_query) {
            max_candidates_per_query = h_num_samples[q];
        }
    }
    
    
    // 计算内存大小
    size_t size_query_group = n_query * n_dim * sizeof(float);
    size_t size_cluster_query_offset = (n_total_clusters + 1) * sizeof(int);  // 标准offset数组格式：n+1个元素
    size_t size_cluster_query_data = n_query * n_cluster * sizeof(int);  // 每个query对应n个cluster
    size_t size_cluster_vector_index = n_total_clusters * sizeof(int);  // distinct cluster数量
    size_t size_cluster_vector_num = n_total_clusters * sizeof(int);  // distinct cluster数量
    size_t size_cluster_vector = n_total_vectors * n_dim * sizeof(float);  // 总向量数量
    size_t size_topn_index = n_query * n_topn * sizeof(int);
    size_t size_topn_dist = n_query * n_topn * sizeof(float);
    
    // 分配设备内存
    float *d_query_group, *d_cluster_vector, *d_topn_dist, *d_query_norm, *d_cluster_vector_norm;
    int *d_cluster_query_offset, *d_cluster_query_data;
    int *d_cluster_vector_index, *d_cluster_vector_num, *d_topn_index, *d_query_mutex;
    
    dim3 clusterDim(n_total_vectors);
    dim3 vectorDim(n_dim);
    dim3 queryDim(n_query);
    
    // GPU内存分配
    cudaMalloc(&d_query_group, size_query_group);
    cudaMalloc(&d_cluster_query_offset, size_cluster_query_offset);
    cudaMalloc(&d_cluster_query_data, size_cluster_query_data);
    cudaMalloc(&d_cluster_vector_index, size_cluster_vector_index);
    cudaMalloc(&d_cluster_vector_num, size_cluster_vector_num);
    cudaMalloc(&d_cluster_vector, size_cluster_vector);
    cudaMalloc(&d_query_norm, n_query * sizeof(float));  // 存储query的L2范数
    cudaMalloc(&d_cluster_vector_norm, n_total_vectors * sizeof(float));  // 存储cluster向量的L2范数
    cudaMalloc(&d_query_mutex, n_query * sizeof(int));  // 每个query一个锁（目前未使用，但保留接口）
    cudaMalloc(&d_topn_index, size_topn_index);
    cudaMalloc(&d_topn_dist, size_topn_dist);
    
    // 复制数据到设备内存
    cudaMemcpy(d_query_group, h_query_group[0], size_query_group, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_query_offset, h_cluster_query_offset, size_cluster_query_offset, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_query_data, h_cluster_query_data, size_cluster_query_data, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vector_index, h_cluster_vector_index, size_cluster_vector_index, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vector_num, h_cluster_vector_num, size_cluster_vector_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vector, h_cluster_vector[0], size_cluster_vector, cudaMemcpyHostToDevice);
    
    // // 初始化锁数组和top-k数组
    // cudaMemset(d_query_mutex, 0, n_query * sizeof(int)); // 锁初始化为0（未锁定）
    // thrust::fill(
    //     thrust::device_pointer_cast(d_topn_dist),
    //     thrust::device_pointer_cast(d_topn_dist) + (n_query * n_topn),
    //     FLT_MAX
    // );
    
    // 使用优化后的warp-sort融合算子
    // 计算cluster向量的L2范数
    {
        CUDATimer timer_compute("Kernel Execution: l2 Norm", ENABLE_CUDA_TIMING);
        // 计算查询向量的L2范数
        l2_norm_kernel<<<queryDim, vectorDim, n_dim * sizeof(float)>>>(
            d_query_group, d_query_norm, 
            n_query, n_dim
        );
        l2_norm_kernel<<<clusterDim, vectorDim, n_dim * sizeof(float)>>>(
            d_cluster_vector, d_cluster_vector_norm, 
            n_total_vectors, n_dim
        );
        cudaDeviceSynchronize();
    }

    // 将num_samples复制到GPU（用于top-k kernel）
    int* d_num_samples;
    cudaMalloc(&d_num_samples, n_query * sizeof(int));
    cudaMemcpy(d_num_samples, h_num_samples, n_query * sizeof(int), cudaMemcpyHostToDevice);
    
    // 调用优化后的融合余弦距离top-k计算
    {
        CUDATimer timer_compute("Kernel Execution: Fusion Cos Top-K", ENABLE_CUDA_TIMING);
        
        cuda_cos_topk_warpsort_fine(
            d_query_group, d_cluster_vector, d_cluster_query_offset, d_cluster_query_data,
            d_cluster_vector_index, d_cluster_vector_num,
            d_query_norm, d_cluster_vector_norm,
            d_topn_index, d_topn_dist,
            n_query, n_total_clusters, n_dim, n_topn, n_total_vectors, 
            max_candidates_per_query, d_num_samples
        );
        
        cudaDeviceSynchronize();
        CHECK_CUDA_ERRORS;
    }

    // 复制结果回主机内存
    cudaMemcpy(h_query_topn_index[0], d_topn_index, size_topn_index, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_query_topn_dist[0], d_topn_dist, size_topn_dist, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;
    
    // 释放设备内存
    cudaFree(d_query_group);
    cudaFree(d_cluster_query_offset);
    cudaFree(d_cluster_query_data);
    cudaFree(d_cluster_vector_index);
    cudaFree(d_cluster_vector_num);
    cudaFree(d_cluster_vector);
    cudaFree(d_query_norm);
    cudaFree(d_cluster_vector_norm);
    cudaFree(d_query_mutex);
    cudaFree(d_topn_index);
    cudaFree(d_topn_dist);
    cudaFree(d_num_samples);
    
    // 释放主机内存
    free(h_num_samples);
}