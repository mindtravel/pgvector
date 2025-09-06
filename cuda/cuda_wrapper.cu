#include <cuda_runtime.h>
#include "cuda_wrapper.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <stdio.h>

/*
* 全局变量用于GPU内存管理
*/ 
static float* d_query_vector = NULL;
static float* d_list_vectors = NULL;
static int* d_list_offsets = NULL;
static int* d_list_counts = NULL;
static float* d_distances = NULL;
static int* d_indices = NULL;
static int gpu_initialized = 0;

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
     * GPU向量搜索批处理（l2距离版本）
     **/ 
    int gpu_ivf_search_l2_batch(
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
        // TODO: 
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
        
        //TODO:

        return 0;
    }
    
}