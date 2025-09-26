/**
 * kernels.cu
 * cuda核函数
 */
#include "kernels.h"
#include <device_launch_parameters.h>
// #include <thrust/device_ptr.h>
// #include <thrust/reduce.h>
// #include <thrust/sort.h>
// #include <thrust/sequence.h>
// #include <thrust/transform.h>
// #include <thrust/functional.h>
// #include <thrust/execution_policy.h>
#include "pch.h"

/*
* 计算模长
* Args:
*   vectors: 原始向量组
*   vetcor_suqared_sum: 规范化后向量组的l2 norm
*   n_dim: 向量维数 
*   n_batch: 一组中向量个数 
*  
* 在向量的分量上并行计算元素的平方，再使用树形规约，最后sqrt
* 速度更快但是无法保留原始向量信息
*/
__global__ void l2_norm_kernel(float *vectors, float *vector_l2_norm, int n_batch, int n_dim) {
    extern __shared__ float shared_mem[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * n_dim + tid;

    // 计算当前向量的平方和
    float square = 0.0f;
    if (tid < n_dim) {
        square = vectors[idx] * vectors[idx];
        // printf("%d: %f %f\n", bid, vectors[idx], square);
    }

    shared_mem[tid] = square;
    __syncthreads();

    // 规约求和
    for (int s = n_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
            // int partner_idx = tid + s;
            // if (partner_idx < n_dim) { // 确保不越界
            //     shared_mem[tid] += shared_mem[partner_idx];
            // }
        }
        __syncthreads();
    }
    
    // 计算L2范数
    if (tid == 0) {
        vector_l2_norm[bid] = sqrt(shared_mem[0]);
        // printf("%d: %f\n", bid, vector_l2_norm[bid]);
    }
}
    

/*
* 按模长归一化
* Args:
*   vectors: 原始向量组
*   vetcor_suqared_sum: 规范化后向量组的l2 norm
*   n_dim: 向量维数 
*   n_batch: 一组中向量个数 
* 归一化
*/
__global__ void normalize_kernel(float *vectors, float* vector_norms, int n_batch, int n_dim) {
    int bid = blockIdx.x;
    // int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // int pos_idx = idx % n_dim;
    // int vector_idx = idx / n_dim;

    /*全零向量不做normalization*/
    if(vector_norms[bid] != 0)
        vectors[idx] /= vector_norms[bid];

}

// /*
// * 计算余弦距离
// * Args:
// *   query_vector: query向量组
// *   data_vector: data向量组
// *   d_cos_dist: cos距离矩阵
// *   n_query: query向量个数
// *   n_dim: 向量维数 
// *   n_batch: 一组中向量个数 
// * 归一化
// */
// __global__ void cos_distance_kernel(
//     float *query_l2_norm, 
//     float *data_l2_norm, 
//     float *d_cos_dist, 
//     int n_query, int n_batch, int n_dim
// ) {
//     int data_id = blockIdx.x;
//     int query_id = threadIdx.x;
//     int idx = data_id * n_query + query_id;
    
//     // 边界检查
//     if (query_id >= n_query || data_id >= n_batch) {
//         // printf("Error: query_id=%d (max %d), data_id=%d (max %d)\n", 
//         //        query_id, n_query-1, data_id, n_batch-1);
//         return;
//     }
    
//     // 检查零向量情况
//     float query_norm = query_l2_norm[query_id];
//     float data_norm = data_l2_norm[data_id];
    
//     // printf("Error: query_norm=%f, data_norm=%f\n", 
//         // query_norm, data_norm);
//     if (query_norm < 1e-6f || data_norm < 1e-6f) {
//         d_cos_dist[idx] = 0.0f;  // 如果任一向量接近零向量，相似度为0
//     } else {
//         printf("cos_dist query=%d, data=%d, dist=%f\n", query_id, data_id, d_cos_dist[idx]);
//         d_cos_dist[idx] /= (query_norm * data_norm);
//         // printf("cos_dist query=%d, data=%d, dist=%f\n", query_id, data_id, d_cos_dist[idx]);
//     }
// }
__global__ void cos_distance_kernel(
    float *query_l2_norm, float *data_l2_norm, float *d_cos_dist, 
    int n_query, int n_batch, int n_dim
) {
    int data_id = blockIdx.x;
    int query_id = threadIdx.x;
    // int idx = data_id * n_query + query_id;
    int idx = data_id + query_id * n_batch;

    // 边界检查 + 错误防护
    if (query_id >= n_query || data_id >= n_batch || idx >= n_batch * n_query) {
        #ifdef DEBUG
        printf("Boundary Error: query_id=%d/%d, data_id=%d/%d\n", 
               query_id, n_query, data_id, n_batch);
        #endif
        return;
    }

    // // 共享内存缓存query范数
    // __shared__ float s_query_norm[256];
    // if (threadIdx.y == 0 && threadIdx.x < n_query) {
    //     s_query_norm[threadIdx.x] = query_l2_norm[threadIdx.x];
    // }
    // __syncthreads();

    // float query_norm = s_query_norm[query_id % 256]; // 假设BLOCK_SIZE=256
    float query_norm = query_l2_norm[query_id]; // 假设BLOCK_SIZE=256
    float data_norm = data_l2_norm[data_id];

    // 范数合法性检查
    if (!isfinite(query_norm) || !isfinite(data_norm) || 
        fabsf(query_norm) < 1e-6f || fabsf(data_norm) < 1e-6f) {
        d_cos_dist[idx] = 0.0f;
        return;
    }

    // 计算归一化余弦距离
    d_cos_dist[idx] /= (query_norm * data_norm);
    // d_cos_dist[idx] = (query_norm);
}

__global__ void topk_kernel(
    int *d_index,
    float *d_dist,
    int *d_topk_index,
    float *d_topk_dist,
    int n_query, int n_batch, int k
){
    /**
     * 动态为每个query维护一个最小top_k
     * 在GPU上进行排序操作
     * 
     * Args:
     * int *d_index    （当前）batch对应向量的索引，形状为 [n_batch]
     * float *d_dist        距离矩阵，形状为 [n_query, n_batch]
     * int *d_topk_index    （当前）K近邻索引，形状为 [n_query, k]
     * float *d_topk_dist   （当前）k近邻距离，形状为 [n_query, k]
     * int n_query          query数量
     * int n_batch          一个batch的向量个数
     * int k                近邻个数k
     */

}