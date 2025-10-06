#include "fusion_cos_topk.cuh"

// 定义最大k值，用于共享内存分配
#define MAX_K 1024


__device__ void mutex_lock(int* mutex) {
    /**
     * 互斥锁加锁
     **/
    while (atomicExch(mutex, 1) == 1) {
        /* 自旋等待 */ 
    }
}

__device__ void mutex_unlock(int* mutex) {
    /**
     * 互斥锁解锁
     **/
    atomicExch(mutex, 0);
}


__device__ void shared_heap_insert(float* heap_dist, int* heap_idx, float dist, int idx, int* heap_size, int k) {
    /**
     * 向共享内存堆插入新元素
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

__device__ void shared_heap_replace_max(float* heap_dist, int* heap_idx, float dist, int idx, int k) {
    /**
    * 替换共享内存最大堆的根节点，并调整堆
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

__global__ void fusion_cos_topk_kernel(
    float* d_query_norm, float* d_data_norm, float* d_inner_product, int* d_index,
    int* topk_index, float* topk_dist,
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
    * 初版：
    *     单次规约，堆的具体功能在共享内存中实现，实现完毕后复制到HBM内存中
    * 
    * 接下来的改进：
    *     两次规约，按照共享内存的大小划分block，每个block维护一个候选队列，从对应topk距离中读到topk中的最大值，这个候选队列将所有小于"最大值"的距离和索引都保存下来
    *     然后将候选队列中的数替换最大值，同时更新堆
    **/
    
    int query_idx = blockIdx.x;
    int data_idx = threadIdx.x;
    
    /**
     * 共享内存中的堆数据，每个block维护一个query的topk堆 
     **/ 
    extern __shared__ float shared_heap_dist[];  /* 动态共享内存（需要学习具体是怎么工作的） */ 
    __shared__ int* shared_heap_idx;  /* 指向共享内存中的索引数组 */ 
    __shared__ int heap_mutex; /* 堆中的互斥锁 */
    __shared__ int heap_size;  /* 当前堆中元素数量 */ 

    if (query_idx >= n_query || data_idx >= n_batch) return;

    /**
     * 初始化共享内存和互斥锁（只有第一个线程执行）
     **/ 
    if (threadIdx.x == 0) {
        heap_mutex = 0;
        heap_size = 0;
        shared_heap_idx = (int*)(shared_heap_dist + k);  /* 索引数组紧跟在距离数组后面 */ 
    }
    __syncthreads();
    
    /**
     * 获取当前query对应的全局堆（用于最终结果）
     **/ 
    float* global_heap_dist = &topk_dist[query_idx * k];
    int* global_heap_idx = &topk_index[query_idx * k];

    /**
     * HBM内存中读取计算需要的数据，访问HBM只需要这么多次数，
     * 这方面的性能应该是不错的
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

    float cosine_similarity = inner_product / (query_norm * data_norm);
    float cosine_distance = 1.0f - cosine_similarity;
    // printf("%d %d %f\n", query_idx, data_idx, cosine_distance);



    /**
     * 使用互斥锁确保线程安全地更新共享内存中的堆
     **/ 
    mutex_lock(&heap_mutex);
    
    if (heap_size < k) {
        /* 堆未满，直接插入 */ 
        shared_heap_insert(shared_heap_dist, shared_heap_idx, cosine_distance, index, &heap_size, k);
    } else {
        /* 堆已满，检查是否需要替换最大值 */ 
        float current_max = shared_heap_dist[0];  /* 最大堆的根节点是最大值 */
        if (cosine_distance < current_max) {
            /* 替换最大值，并且调整堆 */ 
            shared_heap_replace_max(shared_heap_dist, shared_heap_idx, cosine_distance, index, k);
        }
    }
    
    mutex_unlock(&heap_mutex);
    
    __syncthreads();
    
    /**
     * 将共享内存中的结果复制到全局内存（只有第一个线程执行）
     **/ 
    if (threadIdx.x == 0) {
        for (int i = 0; i < heap_size && i < k; i++) {
            global_heap_dist[i] = shared_heap_dist[i];
            global_heap_idx[i] = shared_heap_idx[i];
        }
    }
}