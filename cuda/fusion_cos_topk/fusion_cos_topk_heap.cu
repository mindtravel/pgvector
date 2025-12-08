#include "../utils.cuh"
#include <cfloat>

// #include ".h"

#include "fusion_cos_topk.cuh"
#include "../l2norm/l2norm.cuh"
#include "../unit_tests/common/test_utils.cuh"
#include "pch.h"

// 定义最大k值
#define MAX_K 1024
#define ENABLE_CUDA_TIMING 1 /*是否启用CUDATimer计时*/

// mutex_lock和mutex_unlock已在fusion_cos_topk.cuh中定义为inline

__device__ void global_heap_insert(float* heap_dist, int* heap_idx, float dist, int idx, int* heap_size, int k) {
    /**
     * 向普通显存堆插入新元素
     **/ 
    int pos = *heap_size;
    if (pos >= k) return;  /* 堆已满 */ 
    
    heap_dist[pos] = dist;
    heap_idx[pos] = idx;
    (*heap_size)++; 
    
    /* 自底向上调整堆 */ 
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (heap_dist[parent] >= heap_dist[pos]) break;
        
        /* 交换 */ 
        float temp_dist = heap_dist[parent];
        int temp_idx = heap_idx[parent];
        heap_dist[parent] = heap_dist[pos];
        heap_idx[parent] = heap_idx[pos];
        heap_dist[pos] = temp_dist;
        heap_idx[pos] = temp_idx;
        
        pos = parent;
    }
}

__device__ void global_heap_replace_max(float* heap_dist, int* heap_idx, float dist, int idx, int k) {
    /**
    * 替换普通显存最大堆的根节点，并调整堆
    **/ 
    heap_dist[0] = dist;
    heap_idx[0] = idx;
    
    /* 自顶向下调整堆 */
    int pos = 0;
    while (true) {
        int left = 2 * pos + 1;
        int right = 2 * pos + 2;
        int largest = pos;

        if (left < k && heap_dist[left] > heap_dist[largest]) {
            largest = left;
        }
        if (right < k && heap_dist[right] > heap_dist[largest]) {
            largest = right;
        }
        
        if (largest == pos) break;
        
        /* 交换 */ 
        float temp_dist = heap_dist[pos];
        int temp_idx = heap_idx[pos];
        heap_dist[pos] = heap_dist[largest];
        heap_idx[pos] = heap_idx[largest];
        heap_dist[largest] = temp_dist;
        heap_idx[largest] = temp_idx;
        
        pos = largest;
    }
}

__global__ void fusion_cos_topk_heap_kernel(
    float* d_query_norm, float* d_data_norm, float* d_inner_product, int* d_index,
    int* topk_index, float* topk_dist,
    int* heap_mutexes, int* heap_sizes,
    int n_query, int n_batch, int k
){
    /**
    * 余弦距离和topk融合算子
    * 
    * 目的：在现有查询中，每个query和n_batch个data向量两两内积，得到 d_inner_product（大小为[n_query, n_batch]）
    * 每个query的L2范数存在d_query_norm（大小为[n_query]）
    * 每个data的L2范数存在d_data_norm（大小为[n_batch]）
    * 我们需要为每个query向量维护和data距离最小的topk个索引和距离
    * 索引存在topk_index（大小为[n_query, k]）
    * 距离存在topk_dist（大小为[n_query, k]）
    * 
    * 具体实现：
    *     线程模型为<<<n_query, n_batch>>>
    *     采用流式计算，每个block从query_norm中读取对应的范数，每个thread从data_norm中读取对应的范数，计算inner_product/sqrt(query_norm*data_norm)
    *     topk_dist中的每一行是一个最大堆，其第一个元素是堆中最大值，也是会被替换的数
    * 
    * 改进版：
    *     使用普通显存（global memory）而非共享内存实现堆，直接操作全局内存中的topk数据
    *     堆的互斥锁和大小也存储在全局内存中
    **/
    
    int query_idx = blockIdx.x;
    int data_idx = threadIdx.x;
    
    if (query_idx >= n_query || data_idx >= n_batch) return;
    
    /**
     * 获取当前query对应的全局堆（直接在全局内存中操作）
     **/ 
    float* global_heap_dist = &topk_dist[query_idx * k];
    int* global_heap_idx = &topk_index[query_idx * k];
    int* heap_mutex = &heap_mutexes[query_idx];
    int* heap_size = &heap_sizes[query_idx];

    /**
     * 从全局内存中读取计算需要的数据
     */
    float query_norm = d_query_norm[query_idx];
    float data_norm = d_data_norm[data_idx];
    float inner_product = d_inner_product[query_idx * n_batch + data_idx];
    float index = d_index[query_idx * n_batch + data_idx];
    
    /**
     * 余弦相似度 = inner_product / (query_norm * data_norm)
     * 余弦距离 = 1 - 余弦相似度
     * 
     * 今后可能的调整：看将求平方根放到这一步好还是放到求范数好，
     * 也就是说，在数值精度和计算速度间权衡
     **/ 

    float cos_similarity = inner_product / (query_norm * data_norm);
    float cos_distance = 1.0f - cos_similarity;
    // printf("%d %d %f\n", query_idx, data_idx, cos_distance);



    /**
     * 使用互斥锁确保线程安全地更新全局内存中的堆
     **/ 
    mutex_lock(heap_mutex);
    
    if (*heap_size < k) {
        /* 堆未满，直接插入 */ 
        global_heap_insert(global_heap_dist, global_heap_idx, cos_distance, index, heap_size, k);
    } else {
        /* 堆已满，检查是否需要替换最大值 */ 
        float current_max = global_heap_dist[0];  /* 最大堆的根节点是最大值 */
        if (cos_distance < current_max) {
            /* 替换最大值，并且调整堆 */ 
            global_heap_replace_max(global_heap_dist, global_heap_idx, cos_distance, index, k);
        }
    }
    
    mutex_unlock(heap_mutex);
}


void cuda_cos_topk_heap(
    float** h_query_vectors, float** h_data_vectors, 
    int** h_index, int** h_topk_index, float** h_topk_cos_dist,
    int n_query, int n_batch, int n_dim,
    int k /*查找的最近邻个数*/
){ 
    /**
    * 对一个batch的查询向量，找出余弦距离最近的topk，返回一个形状为 [batch, k] 的索引矩阵
    **/
//    table_2D("h_topk_index", h_topk_index, n_query, k);

    float alpha = 1.0f; 
    float beta = 0.0f;

    const int NUM_STREAMS = 0; // cuda流的数量
    bool query_copied = false; // query常驻显存

    dim3 queryDim(n_query);
    dim3 dataDim(n_batch);
    dim3 vectorDim(n_dim);

    cudaStream_t streams[NUM_STREAMS];
    
    size_t size_query = n_query * n_dim * sizeof(float);
    size_t size_data = n_batch * n_dim * sizeof(float);
    size_t size_dist = n_query * n_batch * sizeof(float);
    size_t size_index = n_query * n_batch * sizeof(int);
    size_t size_topk_dist = n_query * k * sizeof(float);
    size_t size_topk_idx = n_query * k * sizeof(int);

    // cuBLAS句柄
    cublasHandle_t handle;
    // cublasSetStream(handle, streams[0]); 

    // 分配设备内存
    float *d_query_vectors, *d_data_vectors, *d_inner_product, *d_topk_cos_dist,
        *d_query_norm, *d_data_norm;
    int *d_index, *d_topk_index;
    int *d_heap_mutexes, *d_heap_sizes;  /* 全局内存中的堆互斥锁和堆大小数组 */
    {
        CUDATimer timer_manage("GPU Memory Allocation", ENABLE_CUDA_TIMING);

        cudaMalloc(&d_query_vectors, size_query);
        cudaMalloc(&d_data_vectors, size_data);
        cudaMalloc(&d_inner_product, size_dist);/*存储各个query需要查找的data向量的距离*/
        cudaMalloc(&d_index, size_index);/*存储各个query需要查找的data向量的索引*/
        cudaMalloc(&d_topk_cos_dist, size_topk_dist);/*存储topk距离*/
        cudaMalloc(&d_topk_index, size_topk_idx);/*存储topk索引*/

        cudaMalloc(&d_query_norm, n_query * sizeof(float)); /*存储query的l2 Norm*/
        cudaMalloc(&d_data_norm, n_batch * sizeof(float)); /*存储data的l2 Norm*/
        
        cudaMalloc(&d_heap_mutexes, n_query * sizeof(int)); /*存储每个query的堆互斥锁*/
        cudaMalloc(&d_heap_sizes, n_query * sizeof(int)); /*存储每个query的堆大小*/

        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        }

        cublasCreate(&handle);
    }

    // 复制数据到设备
    {

        CUDATimer timer_trans1("H2D Data Transfer", ENABLE_CUDA_TIMING);
        // 复制查询向量，然后常驻
        if(query_copied == false){
            cudaMemcpy2D(
                d_query_vectors,
                n_dim * sizeof(float),
                h_query_vectors[0],
                n_dim * sizeof(float),
                n_dim * sizeof(float),
                n_query,
                cudaMemcpyHostToDevice
            );            
            query_copied = true;
        }

        /* 复制data向量 */
        cudaMemcpy2D(
            d_data_vectors,
            n_dim * sizeof(float),
            h_data_vectors[0],
            n_dim * sizeof(float),
            n_dim * sizeof(float),
            n_batch,
            cudaMemcpyHostToDevice
        );
        // cudaMemcpy(d_data_vectors, h_data_vectors, size_data, cudaMemcpyHostToDevice);        

        /* 复制索引数组 */
        cudaMemcpy2D(
            d_index,
            n_batch * sizeof(int),
            h_index[0],
            n_batch * sizeof(int),
            n_batch * sizeof(int),
            n_query,
            cudaMemcpyHostToDevice
        );
        // CHECK_CUDA_ERRORS;

        /* 初始化距离数组（使用fill kernel替代thrust::fill） */
        dim3 fill_block(256);
        int fill_grid_size = (n_query * k + fill_block.x - 1) / fill_block.x;
        dim3 fill_grid(fill_grid_size);
        fill_kernel<<<fill_grid, fill_block>>>(
            d_topk_cos_dist,
            FLT_MAX,
            n_query * k
        );
        // cudaMemset((void*)d_topk_cos_dist, (int)0xEF, n_query * k * sizeof(float)) /*也可以投机取巧用memset，正好将数组为一个非常大的负数*/
        // table_cuda_2D("topk cos distance", d_topk_cos_dist, n_query, k);

        /* 初始化堆互斥锁和堆大小数组 */
        cudaMemset(d_heap_mutexes, 0, n_query * sizeof(int)); /*互斥锁初始化为0（未锁定）*/
        cudaMemset(d_heap_sizes, 0, n_query * sizeof(int)); /*堆大小初始化为0（空堆）*/

    }

    // print_cuda_2D("index matrix", d_index, n_query, n_batch);
    // print_cuda_2D("cos distance matrix", d_inner_product, n_query, n_batch);
    // print_2D("query vector", h_query_vectors, n_query, n_dim);
    // print_2D("data vector", h_data_vectors, n_batch, n_dim);
    // print_cuda_2D("query vector", d_query_vectors, n_query, n_dim);
    // print_cuda_2D("data vector", d_data_vectors, n_batch, n_dim);
    // print_cuda_2D("topk index matrix", d_topk_index, n_query, k);
    // print_cuda_2D("topk cos distance matrix", d_topk_cos_dist, n_query, k);

    /* 核函数执行 */
    {
        CUDATimer timer_compute("Kernel Execution: l2 Norm + matrix multiply", ENABLE_CUDA_TIMING);

        l2_norm_kernel<<<queryDim, vectorDim, n_dim * sizeof(float)>>>(
            d_query_vectors, d_query_norm, 
            n_query, n_dim
        );

        l2_norm_kernel<<<dataDim, vectorDim, n_dim * sizeof(float)>>>(
            d_data_vectors, d_data_norm, 
            n_batch, n_dim
        );        

        // table_cuda_2D("topk cos distance", d_topk_cos_dist, n_query, k);
       
        // table_cuda_1D("query_norm", d_query_norm, n_query);
        // table_cuda_1D("data_norm", d_data_norm, n_batch);
        // table_cuda_2D("data vectors", d_data_vectors, n_batch, n_dim);

        /**
        * 使用cuBLAS进行矩阵乘法
        * cuBLAS默认使用列主序，leading dimension是行数
        * */ 
       cublasSgemm(handle, 
            CUBLAS_OP_T, CUBLAS_OP_N, 
            n_batch, n_query, n_dim,                   
            &alpha, 
            d_data_vectors, n_dim,            
            d_query_vectors, n_dim,               
            &beta, 
            d_inner_product, n_batch
        );    
        
        cudaDeviceSynchronize(); 

        // print_cuda_2D("inner product", d_inner_product, n_query, n_batch);

        // table_cuda_2D("topk index", d_topk_index, n_query, k);
        // table_cuda_2D("topk cos distance", d_topk_cos_dist, n_query, k);
    }

    {
        CUDATimer timer_compute("Kernel Execution: cos + topk", ENABLE_CUDA_TIMING);

        fusion_cos_topk_heap_kernel<<<queryDim, dataDim>>>(
            d_query_norm, d_data_norm, d_inner_product, d_index,
            d_topk_index, d_topk_cos_dist,
            d_heap_mutexes, d_heap_sizes,
            n_query, n_batch, k
        );

        // table_cuda_2D("topk index", d_topk_index, n_query, k);
        // table_cuda_2D("topk cos distance", d_topk_cos_dist, n_query, k);

        cudaDeviceSynchronize(); 
        
        // 设置CUDA设备以确保RAFT库使用正确的设备
        

    }


    {
        CUDATimer timer_trans2("D2H Data Transfer", ENABLE_CUDA_TIMING);
        cudaMemcpy(h_topk_index[0], d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost);        
        cudaMemcpy(h_topk_cos_dist[0], d_topk_cos_dist, n_query * k * sizeof(float), cudaMemcpyDeviceToHost);        
    }

    {
        CUDATimer timer_manage2("GPU Memory Free", ENABLE_CUDA_TIMING, false);
        cublasDestroy(handle);
        cudaFree(d_query_vectors);
        cudaFree(d_data_vectors);
        cudaFree(d_inner_product);
        cudaFree(d_query_norm);
        cudaFree(d_data_norm);
        cudaFree(d_index);
        cudaFree(d_topk_cos_dist);
        cudaFree(d_topk_index);
        cudaFree(d_heap_mutexes);
        cudaFree(d_heap_sizes);
        
        // 销毁CUDA流
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }

    // CHECK_CUDA_ERRORS;
}