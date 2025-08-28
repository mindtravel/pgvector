#include <cuda_runtime.h>
#include "cuda_wrapper.h"
#include "cuda_distances.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <stdio.h>

/**
 * 包装C语言调用cuda函数的接口
 **/
extern "C" {
    // 检查CUDA是否可用
    bool cuda_is_available() {
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        return (err == cudaSuccess && device_count > 0);
    }

    /**
     * GPU向量搜索初始化
     **/ 
    bool gpu_ivf_search_init() {
        if (!cuda_is_available()) return false;
        gpu_initialized = 1;
        /* 设置GPU */
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            return false;
        }
        
        return true;
    }

    /**
     * GPU向量搜索清理
     **/ 
    void gpu_ivf_search_cleanup() {
        /* 清理查询资源 */
        if (d_query_vector) {
            cudaFree(d_query_vector);
            d_query_vector = NULL;
        }
        if (d_list_vectors) {
            cudaFree(d_list_vectors);
            d_list_vectors = NULL;
        }
        if (d_list_offsets) {
            cudaFree(d_list_offsets);
            d_list_offsets = NULL;
        }
        if (d_list_counts) {
            cudaFree(d_list_counts);
            d_list_counts = NULL;
        }
        if (d_distances) {
            cudaFree(d_distances);
            d_distances = NULL;
        }
        if (d_indices) {
            cudaFree(d_indices);
            d_indices = NULL;
        }
        
        gpu_initialized = 0;
        /* 同步设备 */
        cudaDeviceSynchronize();
    }

    /**
     * GPU向量搜索批处理（基础版本）
     **/ 
    int gpu_ivf_search_batch(
        float* query_vector,
        float* list_vectors,
        int* list_offsets,
        int* list_counts,
        int num_lists,
        int vector_dim,
        float* distances,
        int* indices,
        int k
    ) {
        // 基础实现 - 暂时返回错误
        // 后续步骤会完善这个函数
        return -1;
    }

    /**
     * GPU向量搜索批处理（余弦版本）
     **/ 
    int gpu_ivf_search_cosine_batch(
        float* query_vector,
        float* list_vectors,
        int* list_offsets,
        int* list_counts,
        int num_lists,
        int vector_dim,
        float* distances,
        int* indices,
        int k
    ) {
        if (!gpu_initialized) return -1;
        
        // 计算总向量数
        int total_vectors = 0;
        for (int i = 0; i < num_lists; i++) {
            total_vectors += list_counts[i];
        }
        
        if (total_vectors == 0) return 0;
        
        // 分配GPU内存（如果需要）
        size_t query_size = vector_dim * sizeof(float);
        size_t vectors_size = total_vectors * vector_dim * sizeof(float);
        size_t offsets_size = num_lists * sizeof(int);
        size_t counts_size = num_lists * sizeof(int);
        size_t distances_size = total_vectors * sizeof(float);
        size_t indices_size = total_vectors * sizeof(int);
        
        // 分配或重新分配GPU内存
        if (!d_query_vector || !d_list_vectors || !d_list_offsets || 
            !d_list_counts || !d_distances || !d_indices) {
            gpu_ivf_search_cleanup();
            
            cudaMalloc(&d_query_vector, query_size);
            cudaMalloc(&d_list_vectors, vectors_size);
            cudaMalloc(&d_list_offsets, offsets_size);
            cudaMalloc(&d_list_counts, counts_size);
            cudaMalloc(&d_distances, distances_size);
            cudaMalloc(&d_indices, indices_size);
        }
        
        // 传输数据到GPU
        cudaMemcpy(d_query_vector, query_vector, query_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_list_vectors, list_vectors, vectors_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_list_offsets, list_offsets, offsets_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_list_counts, list_counts, counts_size, cudaMemcpyHostToDevice);
        
        // 初始化索引数组
        thrust::sequence(thrust::device, d_indices, d_indices + total_vectors);
        
        // 计算距离
        int block_size = 256;
        int grid_size = (total_vectors + block_size - 1) / block_size;
        compute_cosine_distances_kernel<<<grid_size, block_size>>>(
            d_query_vector, d_list_vectors, d_list_offsets, d_list_counts,
            d_distances, vector_dim, total_vectors
        );
        
        // 使用thrust进行排序，选择前k个最小距离
        thrust::sort_by_key(thrust::device, d_distances, d_distances + total_vectors, d_indices);
        
        // 将结果传回CPU
        cudaMemcpy(distances, d_distances, k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(indices, d_indices, k * sizeof(int), cudaMemcpyDeviceToHost);
        
        return k;
    }
    
}