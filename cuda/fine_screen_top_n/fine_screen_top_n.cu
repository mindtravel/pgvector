#include "fine_screen_top_n.cuh"
#include "../l2norm.cuh"
#include "../unit_tests/common/test_utils.cuh"
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <limits.h>

#define ENABLE_CUDA_TIMING 0

/**
 * 预留的warpsort接口，用于在寄存器中维护局部top-k
 * 参数待定，实现待定
 */
__device__ void cluster_warpsort_topk(
    float* local_distances,    // 当前cluster的距离数组
    int* local_indices,        // 对应的索引数组
    int cluster_vector_count,   // 当前cluster的向量数量
    int k,                     // top-k数量
    float* output_distances,   // 输出距离
    int* output_indices        // 输出索引
) {
    // 实现待定
    // 这里暂时用简单的排序实现
    for (int i = 0; i < cluster_vector_count - 1; i++) {
        for (int j = i + 1; j < cluster_vector_count; j++) {
            if (local_distances[i] > local_distances[j]) {
                // 交换距离
                float temp_dist = local_distances[i];
                local_distances[i] = local_distances[j];
                local_distances[j] = temp_dist;
                
                // 交换索引
                int temp_idx = local_indices[i];
                local_indices[i] = local_indices[j];
                local_indices[j] = temp_idx;
            }
        }
    }
    
    // 复制前k个结果
    for (int i = 0; i < k && i < cluster_vector_count; i++) {
        output_distances[i] = local_distances[i];
        output_indices[i] = local_indices[i];
    }
}

/**
 * 计算cluster中向量与query的L2距离并选择top-k
 * 每个block处理一个cluster
 */
__global__ void cluster_l2_distance_kernel(
    const float* __restrict__ d_query_group,
    const float* __restrict__ d_query_norm,
    const float* __restrict__ d_cluster_vector,
    const float* __restrict__ d_cluster_vector_norm,
    const int* __restrict__ d_query_cluster_group,
    const int* __restrict__ d_cluster_query_offset,
    const int* __restrict__ d_cluster_query_data,
    const int* __restrict__ d_cluster_map,
    const int* __restrict__ d_cluster_vector_index,
    const int* __restrict__ d_cluster_vector_num,
    int n_query, int n_cluster, int n_dim, int n_topn,
    int max_cluster_vector_count, int distinct_cluster_count,
    int* __restrict__ d_query_mutex,
    int* __restrict__ d_topn_index,
    float* __restrict__ d_topn_dist
) {
    int cluster_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    if (cluster_idx >= distinct_cluster_count || thread_idx >= d_cluster_vector_num[cluster_idx]) return;
    // 共享内存：缓存L2范数和cluster向量数据
    extern __shared__ float shared_mem[];
    float* s_query_norm = shared_mem;
    float* s_cluster_norm = s_query_norm + n_query;
    
    // 只有第一个线程计算query范围，避免越界
    int query_start, query_count;
    if (thread_idx == 0) {
        // 边界检查：确保不越界访问
        if (cluster_idx >= distinct_cluster_count) {
            query_count = 0;
        } else {
            query_start = d_cluster_query_offset[cluster_idx];
            
            // 对于最后一个cluster，使用总数作为结束位置
            if (cluster_idx + 1 >= distinct_cluster_count) {
                // 最后一个cluster：query_count = 总query数 - query_start
                query_count = n_query - query_start;
            } else {
                query_count = d_cluster_query_offset[cluster_idx + 1] - query_start;
            }
            
            // 额外的越界检查
            if (query_start >= n_query || query_start + query_count > n_query || query_count < 0) {
                query_count = 0;
            }
        }
    }
    if (query_count == 0) return;
    __syncthreads();
    
    
    // 加载L2范数到共享内存
    if (thread_idx < n_query) {
        s_query_norm[thread_idx] = d_query_norm[thread_idx];
    }
    if (thread_idx < max_cluster_vector_count) {
        s_cluster_norm[thread_idx] = d_cluster_vector_norm[thread_idx];
    }
    __syncthreads();
    
    // 获取当前cluster的向量信息
    int vector_start_idx = d_cluster_vector_index[cluster_idx];
    int vector_count = d_cluster_vector_num[cluster_idx];
    
    // 每个线程处理cluster中的部分向量
    int vectors_per_thread = (vector_count + blockDim.x - 1) / blockDim.x;
    int start_vec = thread_idx * vectors_per_thread;
    int end_vec = min(start_vec + vectors_per_thread, vector_count);
    // 每个线程维护局部topk结果
    float local_topk_dist[n_topn];
    int local_topk_idx[n_topn];
    
    // 初始化局部topk为最大值
    for (int i = 0; i < n_topn; i++) {
        local_topk_dist[i] = FLT_MAX;
        local_topk_idx[i] = -1;
    }
    
    // 为每个query计算L2距离并维护局部topk
    for (int q = 0; q < query_count; q++) {
        int query_idx = query_start + q;
        
        // 为当前query维护局部topk
        float query_local_topk_dist[n_topn];
        int query_local_topk_idx[n_topn];
        
        // 初始化当前query的局部topk
        for (int i = 0; i < n_topn; i++) {
            query_local_topk_dist[i] = FLT_MAX;
            query_local_topk_idx[i] = -1;
        }
        
        // 计算当前query与cluster中向量的L2距离
        for (int vec_idx = start_vec; vec_idx < end_vec; vec_idx++) {
            int global_vec_idx = vector_start_idx + vec_idx;
            
            // 计算L2距离的平方（使用L2范数优化）    todo 其实这里也可以提前计算出来 后续看哪个性能更好一点吧
            float dot_product = 0.0f;
            // for (int dim = 0; dim < n_dim; dim++) {
            //     dot_product += d_query_group[query_idx * n_dim + dim] * 
            //                   d_cluster_vector[global_vec_idx * n_dim + dim];
            // }
            
            // L2距离平方 = ||q||^2 + ||v||^2 - 2*q·v
            float distance_squared = s_query_norm[query_idx] + s_cluster_norm[vec_idx] - 2.0f * dot_product;
            
            // 取平方根得到实际距离
            float distance = sqrtf(fmaxf(0.0f, distance_squared));
            
            // // 插入到当前query的局部topk中
            // for (int k = 0; k < n_topn; k++) {
            //     if (distance < query_local_topk_dist[k]) {
            //         // 向后移动元素
            //         for (int m = n_topn - 1; m > k; m--) {
            //             query_local_topk_dist[m] = query_local_topk_dist[m-1];
            //             query_local_topk_idx[m] = query_local_topk_idx[m-1];
            //         }
            //         // 插入新元素
            //         query_local_topk_dist[k] = distance;
            //         query_local_topk_idx[k] = global_vec_idx;
            //         break;
            //     }
            // }
        }
        
    }
    
    __syncthreads();
    
    // 写入显存对应位置 - 使用原子操作加锁
    // 每个线程处理自己负责的query范围
    
    int queries_per_thread = (query_count + blockDim.x - 1) / blockDim.x;
    int start_query = thread_idx * queries_per_thread;
    int end_query = min(start_query + queries_per_thread, query_count);
    
    for (int q = start_query; q < end_query; q++) {
        int query_idx = query_start + q;
        
        // 使用原子操作获取锁
        while (atomicCAS(&d_query_mutex[query_idx], 0, 1) != 0) {
            // 自旋等待
        }
        
        // 合并局部topk到全局topk
        // 合入方式待定，先简单的暴力写入当前cluster vector的前n个 + 1验证正确性
        for (int k = 0; k < n_topn; k++) {
            d_topn_index[query_idx * n_topn + k] = vector_start_idx + k;
            d_topn_dist[query_idx * n_topn + k] = local_topk_dist[k];
        }
        
        // 释放锁
        atomicExch(&d_query_mutex[query_idx], 0);
    }

}

void fine_screen_top_n(
    float* h_query_group, int* h_query_cluster_group, int* h_cluster_query_offset, int* h_cluster_query_data,
    int* cluster_map,
    int* h_cluster_vector_index, int* h_cluster_vector_num, float** h_cluster_vector,
    int n_query, int n_cluster, int distinct_cluster_count, int n_dim, int n_topn, int max_cluster_id, int tol_vector,
    int max_cluster_vector_count,  // 新增：最大聚类向量数量
    int* h_query_topn_index, float* h_query_topn_dist
) {
    // 计算内存大小
    size_t size_query_group = n_query * n_dim * sizeof(float);
    size_t size_query_cluster_group = n_query * n_cluster * sizeof(int); //每个query对应n个cluster
    size_t size_cluster_query_offset = distinct_cluster_count * sizeof(int);  // distinct cluster数量
    size_t size_cluster_query_data = n_query * n_cluster * sizeof(int);  // 每个query对应n个cluster
    size_t size_cluster_map = distinct_cluster_count * sizeof(int);  // distinct cluster数量
    size_t size_cluster_vector_index = distinct_cluster_count * sizeof(int);  // distinct cluster数量
    size_t size_cluster_vector_num = distinct_cluster_count * sizeof(int);  // distinct cluster数量
    size_t size_cluster_vector = tol_vector * n_dim * sizeof(float);  // 总向量数量
    size_t size_topn_index = n_query * n_topn * sizeof(int);
    size_t size_topn_dist = n_query * n_topn * sizeof(float);
    
    // 分配设备内存
    float *d_query_group, *d_cluster_vector, *d_topn_dist, *d_query_norm, *d_cluster_vector_norm;
    int *d_query_cluster_group, *d_cluster_query_offset, *d_cluster_query_data;
    int *d_cluster_vector_index, *d_cluster_vector_num, *d_topn_index, *d_cluster_map, *d_query_mutex;
    
    dim3 clusterDim(tol_vector);
    dim3 vectorDim(n_dim);
    dim3 queryDim(n_query);
    // GPU内存分配
    cudaMalloc(&d_query_group, size_query_group);
    cudaMalloc(&d_query_cluster_group, size_query_cluster_group);
    cudaMalloc(&d_cluster_query_offset, size_cluster_query_offset);
    cudaMalloc(&d_cluster_query_data, size_cluster_query_data);
    //cudaMalloc(&d_cluster_map, size_cluster_map);
    cudaMalloc(&d_cluster_vector_index, size_cluster_vector_index);
    cudaMalloc(&d_cluster_vector_num, size_cluster_vector_num);
    cudaMalloc(&d_cluster_vector, size_cluster_vector);
    cudaMalloc(&d_query_norm, n_query * sizeof(float));  // 存储query的L2范数
    cudaMalloc(&d_cluster_vector_norm, tol_vector * sizeof(float));  // 存储cluster向量的L2范数
    cudaMalloc(&d_query_mutex, n_query * sizeof(int));  // 每个query一个锁
    cudaMalloc(&d_topn_index, size_topn_index);
    cudaMalloc(&d_topn_dist, size_topn_dist);
    
    // 复制数据到设备内存
    cudaMemcpy(d_query_group, h_query_group, size_query_group, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_cluster_group, h_query_cluster_group, size_query_cluster_group, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_query_offset, h_cluster_query_offset, size_cluster_query_offset, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_query_data, h_cluster_query_data, size_cluster_query_data, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_map, cluster_map, size_cluster_map, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vector_index, h_cluster_vector_index, size_cluster_vector_index, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_vector_num, h_cluster_vector_num, size_cluster_vector_num, cudaMemcpyHostToDevice);
    // 使用cudaMemcpy2D从二维指针复制cluster向量数据到设备内存
    // h_cluster_vector[i] 指向第i个cluster的向量数据
    cudaMemcpy2D(
        d_cluster_vector,                    // 目标设备内存
        n_dim * sizeof(float),              // 目标行间距
        h_cluster_vector[0],                // 源主机内存（第一个cluster的向量）
        n_dim * sizeof(float),              // 源行间距
        n_dim * sizeof(float),              // 每行字节数
        tol_vector,                          // 行数（cluster数量）
        cudaMemcpyHostToDevice
    );
    
    // 初始化锁数组和top-k数组
    cudaMemset(d_query_mutex, 0, n_query * sizeof(int)); // 锁初始化为0（未锁定）
    thrust::fill(
        thrust::device_pointer_cast(d_topn_dist),
        thrust::device_pointer_cast(d_topn_dist) + (n_query * n_topn),
        FLT_MAX
    );
    
    // TODO: 在这里添加实际的kernel计算逻辑
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
            tol_vector, n_dim
        );
        cudaDeviceSynchronize();
    }

    {
        CUDATimer timer_compute("Kernel Execution: L2 Distance + Top-K", ENABLE_CUDA_TIMING);
        
        // 计算共享内存大小
        size_t shared_mem_size = (n_query + max_cluster_vector_count) * sizeof(float);
        
        // 调用主要的L2距离计算kernel
        dim3 grid(distinct_cluster_count);
        dim3 block(max_cluster_vector_count);
        
        cluster_l2_distance_kernel<<<grid, block, shared_mem_size>>>(
            d_query_group, d_query_norm, d_cluster_vector, d_cluster_vector_norm,
            d_query_cluster_group, d_cluster_query_offset, d_cluster_query_data,
            d_cluster_map, d_cluster_vector_index, d_cluster_vector_num,
            n_query, n_cluster, n_dim, n_topn, max_cluster_vector_count, distinct_cluster_count,
            d_query_mutex, d_topn_index, d_topn_dist
        );
        
        cudaDeviceSynchronize();
    }

    // 复制结果回主机内存
    cudaMemcpy(h_query_topn_index, d_topn_index, size_topn_index, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_query_topn_dist, d_topn_dist, size_topn_dist, cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_query_group);
    cudaFree(d_query_cluster_group);
    cudaFree(d_cluster_query_offset);
    cudaFree(d_cluster_query_data);
    //cudaFree(d_cluster_map);
    cudaFree(d_cluster_vector_index);
    cudaFree(d_cluster_vector_num);
    cudaFree(d_cluster_vector);
    cudaFree(d_query_norm);
    cudaFree(d_cluster_vector_norm);
    cudaFree(d_query_mutex);
    cudaFree(d_topn_index);
    cudaFree(d_topn_dist);
}