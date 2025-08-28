#include <cuda_runtime.h>
#include "cuda_distances.h"
#include <stdio.h>
#include <math.h>
#include <device_launch_parameters.h>

/**
 * 计算余弦距离的核函数
 * Args:
 * query_vector: 查询向量
 * list_vectors: 选中的聚类中的向量
 * list_offset: 和list中数据排布有关?
 * list_counts: 完全不知道?
 * distances: 储存求解出的距离
 * vector_dim: 向量的维数
 * total_vectors: 不太确定? 
 * 
 * 
 * 
 **/
__global__ void compute_cosine_distances_kernel(
    float* query_vector,
    float* list_vectors,
    int* list_offsets,
    int* list_counts,
    float* distances,
    int vector_dim,
    int total_vectors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_vectors) return;
    
    /**
     * 计算余弦距离: 1 - (A·B) / (|A|·|B|)
     **/
    float dot_product;
    float query_norm;
    float vector_norm;
    
    dot_product = 0.0f;
    query_norm = 0.0f;
    vector_norm = 0.0f;
    /**
     * 有待优化，向量内部还可以进一步并行，但是现在实现功能就可以，先看看数据传输开销
     **/
    for (int i = 0; i < vector_dim; i++) {
        float query_val;
        float vector_val;

        query_val = query_vector[i];
        vector_val = list_vectors[tid * vector_dim + i];
        
        dot_product += query_val * vector_val;
        query_norm += query_val * query_val;
        vector_norm += vector_val * vector_val;
    }
    
    query_norm = sqrt(query_norm);
    vector_norm = sqrt(vector_norm);
    
    /**
     * 避免找到零向量
     **/ 
    if (query_norm > 0 && vector_norm > 0) {
        distances[tid] = 1.0f - (dot_product / (query_norm * vector_norm));
    } else {
        distances[tid] = 1.0f; // 最大距离
    }
}

// // 批量GPU搜索（余弦距离）
// extern "C" {
//     int gpu_ivf_search_cosine_batch(
//         float* query_vector,
//         float* list_vectors,
//         int* list_offsets,
//         int* list_counts,
//         int num_lists,
//         int vector_dim,
//         float* distances,
//         int* indices,
//         int k
//     ) {
//         if (!gpu_initialized) return -1;
        
//         // 计算总向量数
//         int total_vectors = 0;
//         for (int i = 0; i < num_lists; i++) {
//             total_vectors += list_counts[i];
//         }
        
//         if (total_vectors == 0) return 0;
        
//         // 分配GPU内存（如果需要）
//         size_t query_size = vector_dim * sizeof(float);
//         size_t vectors_size = total_vectors * vector_dim * sizeof(float);
//         size_t offsets_size = num_lists * sizeof(int);
//         size_t counts_size = num_lists * sizeof(int);
//         size_t distances_size = total_vectors * sizeof(float);
//         size_t indices_size = total_vectors * sizeof(int);
        
//         // 分配或重新分配GPU内存
//         if (!d_query_vector || !d_list_vectors || !d_list_offsets || 
//             !d_list_counts || !d_distances || !d_indices) {
//             gpu_ivf_search_cleanup();
            
//             cudaMalloc(&d_query_vector, query_size);
//             cudaMalloc(&d_list_vectors, vectors_size);
//             cudaMalloc(&d_list_offsets, offsets_size);
//             cudaMalloc(&d_list_counts, counts_size);
//             cudaMalloc(&d_distances, distances_size);
//             cudaMalloc(&d_indices, indices_size);
//         }
        
//         // 传输数据到GPU
//         cudaMemcpy(d_query_vector, query_vector, query_size, cudaMemcpyHostToDevice);
//         cudaMemcpy(d_list_vectors, list_vectors, vectors_size, cudaMemcpyHostToDevice);
//         cudaMemcpy(d_list_offsets, list_offsets, offsets_size, cudaMemcpyHostToDevice);
//         cudaMemcpy(d_list_counts, list_counts, counts_size, cudaMemcpyHostToDevice);
        
//         // 初始化索引数组
//         thrust::sequence(thrust::device, d_indices, d_indices + total_vectors);
        
//         // 计算距离
//         int block_size = 256;
//         int grid_size = (total_vectors + block_size - 1) / block_size;
//         compute_cosine_distances_kernel<<<grid_size, block_size>>>(
//             d_query_vector, d_list_vectors, d_list_offsets, d_list_counts,
//             d_distances, vector_dim, total_vectors
//         );
        
//         // 使用thrust进行排序，选择前k个最小距离
//         thrust::sort_by_key(thrust::device, d_distances, d_distances + total_vectors, d_indices);
        
//         // 将结果传回CPU
//         cudaMemcpy(distances, d_distances, k * sizeof(float), cudaMemcpyDeviceToHost);
//         cudaMemcpy(indices, d_indices, k * sizeof(int), cudaMemcpyDeviceToHost);
        
//         return k;
//     }
// }