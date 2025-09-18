/**
 * kernels.cu
 * cuda核函数
 */
#include "kernels.h"
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <chrono>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include "cudatimer.h"

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
     * 动态维护一个最小top_k
     * 使用Thrust库在GPU上进行排序操作
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
    
    int query_id = blockIdx.x;
    if (query_id >= n_query) return;
    
    // 计算当前query在数组中的偏移量
    int query_offset = query_id * k;
    int dist_offset = query_id * n_batch;
    
    // 创建临时数组用于合并当前topk和新的batch距离
    // 总大小 = k (已有topk) + n_batch (新batch)
    int total_size = k + n_batch;
    
    // 使用共享内存存储临时数据
    // extern __shared__ float shared_data[];
    // float *temp_dist = shared_data;
    // int *temp_index = (int*)(shared_data + total_size);
    
    // 每个线程处理一部分数据
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // // 复制当前topk结果到临时数组
    // for (int i = tid; i < k; i += block_size) {
    //     temp_dist[i] = d_topk_dist[query_offset + i];
    //     temp_index[i] = d_topk_index[query_offset + i];
    // }
    
    // 复制新batch的距离到topk数组的后半部分
    for (int i = tid; i < n_batch; i += block_size) {
        d_topk_dist[k + i] = d_dist[dist_offset + i];
        d_topk_index[k + i] = d_index[i];
    }
    
    __syncthreads();
    
    // 只有第一个线程执行排序操作
    if (tid == 0) {
        // 使用Thrust进行排序
        thrust::device_ptr<float> dist_ptr(d_topk_dist);
        thrust::device_ptr<int> index_ptr(d_topk_index);
        
        // 按距离排序，同时保持索引对应关系
        thrust::sort_by_key(dist_ptr, dist_ptr + total_size, index_ptr);
        
        // // 将排序后的前k个结果复制回全局内存
        // for (int i = 0; i < k; i++) {
        //     d_topk_dist[query_offset + i] = temp_dist[i];
        //     d_topk_index[query_offset + i] = temp_index[i];
        // }
    }
}


























// CUDA核函数：计算两个向量的点积
__global__ void dotProductKernel(const float* a, const float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] * b[idx];
    }
}

// CUDA核函数：计算向量的平方
__global__ void squareKernel(const float* vec, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = vec[idx] * vec[idx];
    }
}

// CUDA规约核函数：块内求和
__global__ void reduceSumKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// CUDA Kernel 计算平方差
__global__ void l2_distance_kernel(const float* A, const float* B, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = A[idx] - B[idx];
        result[idx] = diff * diff;
    }
}

// 辅助函数：调用topk_kernel
void launch_topk_kernel(
    int *d_index,
    float *d_dist,
    int *d_topk_index,
    float *d_topk_dist,
    int n_query, int n_batch, int k,
    cudaStream_t stream = 0
) {
    // 计算共享内存大小
    // 需要存储: (k + n_batch) * sizeof(float) + (k + n_batch) * sizeof(int)
    int total_size = k + n_batch;
    size_t shared_mem_size = total_size * (sizeof(float) + sizeof(int));
    
    // 设置网格和块大小
    dim3 grid(n_query);
    dim3 block(min(256, total_size)); // 使用256个线程或数据大小，取较小值
    
    // 启动kernel
    topk_kernel<<<grid, block, shared_mem_size, stream>>>(
        d_index, d_dist, d_topk_index, d_topk_dist, n_query, n_batch, k
    );
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

