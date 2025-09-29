#include "fusion_cos_topk.cuh"
// 堆操作辅助函数
__device__ void heap_insert(float* heap_dist, int* heap_idx, float dist, int idx, int k) {
    // 在最大堆中插入新元素
    int pos = k - 1;  // 从最后一个位置开始
    heap_dist[pos] = dist;
    heap_idx[pos] = idx;
    
    // 向上调整堆
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (heap_dist[parent] >= heap_dist[pos]) break;
        
        // 交换
        float temp_dist = heap_dist[parent];
        int temp_idx = heap_idx[parent];
        heap_dist[parent] = heap_dist[pos];
        heap_idx[parent] = heap_idx[pos];
        heap_dist[pos] = temp_dist;
        heap_idx[pos] = temp_idx;
        
        pos = parent;
    }
}

__device__ void heap_replace_max(float* heap_dist, int* heap_idx, float dist, int idx, int k) {
    // 替换最大堆的根节点
    heap_dist[0] = dist;
    heap_idx[0] = idx;
    
    // 向下调整堆
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
        
        // 交换
        float temp_dist = heap_dist[pos];
        int temp_idx = heap_idx[pos];
        heap_dist[pos] = heap_dist[largest];
        heap_idx[pos] = heap_idx[largest];
        heap_dist[largest] = temp_dist;
        heap_idx[largest] = temp_idx;
        
        pos = largest;
    }
}

__device__ void heapify(float* heap_dist, int* heap_idx, int k) {
    // 从最后一个非叶子节点开始向下调整
    for (int i = k / 2 - 1; i >= 0; i--) {
        int pos = i;
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
            
            // 交换
            float temp_dist = heap_dist[pos];
            int temp_idx = heap_idx[pos];
            heap_dist[pos] = heap_dist[largest];
            heap_idx[pos] = heap_idx[largest];
            heap_dist[largest] = temp_dist;
            heap_idx[largest] = temp_idx;
            
            pos = largest;
        }
    }
}

__global__ void fusion_cos_topk_kernel(
    float* d_query_norm, float* d_data_norm, float* d_inner_product,
    int* topk_index, float* topk_dist,
    int n_query, int n_batch, int k
){
    /*
    余弦距离和topk融合算子

    目的：在现有查询中，每个query和n_batch个data向量两两内积，得到 d_inner_product（大小为[n_query, n_batch]）
    每个query的L2范数存在d_query_norm（大小为[n_query]）
    每个data的L2范数存在d_data_norm（大小为[n_batch]）
    我们需要为每个query向量维护和data距离最小的topk个索引和距离
    索引存在topk_index（大小为[n_query, k]）
    距离存在topk_dist（大小为[n_query, k]）

    具体实现：线程模型为<<<n_query, n_batch>>>
    采用流式计算，每个block从query_norm中读取对应的范数，每个thread从data_norm中读取对应的范数，计算inner_product/sqrt(query_norm*data_norm)
    topk_dist中的每一行是一个最大堆，其第一个元素是堆中最大值，也是会被替换的数
    每个block维护一个候选队列，从对应topk距离中读到topk中的最大值，这个候选队列将所有小于"最大值"的距离和索引都保存下来
    然后将候选队列中的数替换最大值，同时更新索引
    */
    
    int query_idx = blockIdx.x;
    int data_idx = threadIdx.x;
    
    if (query_idx >= n_query || data_idx >= n_batch) return;
    
    // 计算余弦距离
    float query_norm = d_query_norm[query_idx];
    float data_norm = d_data_norm[data_idx];
    float inner_product = d_inner_product[data_idx * n_query + query_idx];
    
    // 余弦相似度 = inner_product / (query_norm * data_norm)
    // 余弦距离 = 1 - 余弦相似度
    float cosine_similarity = inner_product / (query_norm * data_norm);
    float cosine_distance = 1.0f - cosine_similarity;
    
    // 获取当前query对应的堆
    float* heap_dist = &topk_dist[query_idx * k];
    int* heap_idx = &topk_index[query_idx * k];
    
    // 由于堆已经初始化为最大值，我们只需要检查当前距离是否比堆中的最大值小
    // 使用原子操作来安全地读取和更新堆的最大值
    float current_max = heap_dist[0];  // 最大堆的根节点是最大值
    
    if (cosine_distance < current_max) {
        // 需要更新堆，使用原子操作确保线程安全
        // 使用原子比较和交换来确保只有一个线程能更新
        float old_max = atomicExch(&heap_dist[0], cosine_distance);
        if (old_max == current_max) {
            // 当前线程获得了更新权限，替换最大值
            heap_idx[0] = data_idx;
            // 向下调整堆以维护堆性质
            heap_replace_max(heap_dist, heap_idx, cosine_distance, data_idx, k);
        } else {
            // 其他线程已经更新了，恢复原值
            atomicExch(&heap_dist[0], old_max);
        }
    }
}