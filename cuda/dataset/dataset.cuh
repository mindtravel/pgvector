#ifndef CLUSTER_DATASET_CUH
#define CLUSTER_DATASET_CUH

#include "../../cuda/kmeans/kmeans.cuh"
#include "cpu_array_utils.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <malloc.h>
/**
 * 统一的Cluster数据集结构
 * 使用 ClusterInfo 和连续存储的数据，与实际的搜索接口保持一致
 */
struct ClusterDataset {
    float* reordered_data;      // [n_total_vectors, vector_dim] 连续存储的重排数据
    int* reordered_indices;     // [n_total_vectors] 重排后的索引数组（与reordered_data同步重排）
    float* centroids;           // [cluster_info.k, vector_dim] 聚类中心（CPU内存）
    ClusterInfo cluster_info;   // cluster信息（offsets和counts，其中k为cluster数量）
    int n_total_vectors;        // 总向量数
    int vector_dim;             // 向量维度
    
    /**
     * 初始化ClusterDataset（使用K-means聚类）
     * 
     * @param n_total_vectors 总向量数
     * @param vector_dim 向量维度
     * @param n_clusters cluster数量
     * @param h_objective 可选的输出：K-means目标函数值（可以为nullptr）
     * @param kmeans_iters K-means迭代次数（默认20）
     * @param use_minibatch 是否使用minibatch算法（默认false，使用Lloyd）
     * @param distance_mode 距离类型（默认COSINE_DISTANCE）
     * @param seed 随机种子（默认1234）
     * @param batch_size 批处理大小（默认1<<20）
     * @param device_id GPU设备ID（默认0）
     */
    void init_with_kmeans(
        int n_total_vectors,
        int vector_dim,
        int n_clusters,
        float* h_objective,
        float* h_data = nullptr,
        int kmeans_iters = 20,
        bool use_minibatch = false,
        DistanceType distance_mode = COSINE_DISTANCE,
        unsigned int seed = 1234,
        int batch_size = (1 << 20),
        int device_id = 0
    ) {
        this->n_total_vectors = n_total_vectors;
        this->vector_dim = vector_dim;
        
        // Step 1: 生成原始测试数据（使用多线程随机初始化）
        // 如果传入外部数据，就只做聚类；如果不传入外部数据，就先随机生成数据再做聚类
        size_t data_size = (size_t)n_total_vectors * vector_dim;
        
        float* h_data_ptr = nullptr;  // 用于处理可能分配的内存
        bool need_free_h_data = false;  // 默认不需要释放（因为是外部数据）
        if (!h_data) {
            need_free_h_data = true;  // 如果没有传入数据，我们分配了内存，需要释放
            h_data_ptr = (float*)memalign(64, data_size * sizeof(float));
            init_array_multithreaded(h_data_ptr, data_size, seed, -1.0f, 1.0f);
        } else {
            h_data_ptr = h_data;  // 使用外部传入的数据
        }
        
        // Step 2: 初始化聚类中心
        float* h_init_centroids = nullptr;
        cudaMallocHost(&h_init_centroids, sizeof(float) * (size_t)n_clusters * vector_dim);
        
        KMeansCase kmeans_cfg;
        kmeans_cfg.n = n_total_vectors;
        kmeans_cfg.dim = vector_dim;
        kmeans_cfg.k = n_clusters;
        kmeans_cfg.iters = kmeans_iters;
        kmeans_cfg.minibatch_iters = kmeans_iters * 4; // Minibatch需要更多迭代
        kmeans_cfg.seed = seed;
        kmeans_cfg.dist = distance_mode;
        kmeans_cfg.dtype = USE_FP32;
        
        init_centroids_by_sampling(kmeans_cfg, h_data_ptr, h_init_centroids);
        
        // Step 3: 分配GPU内存
        float* d_centroids = nullptr;
        cudaMalloc(&d_centroids, sizeof(float) * (size_t)n_clusters * vector_dim);
        cudaMemcpy(d_centroids, h_init_centroids, sizeof(float) * (size_t)n_clusters * vector_dim, cudaMemcpyHostToDevice);
        
        // Step 4: 分配重排后的数据缓冲区和索引数组
        reordered_data = (float*)memalign(64, data_size * sizeof(float));
        reordered_indices = (int*)memalign(64, n_total_vectors * sizeof(int));
        int* original_indices = (int*)memalign(64, n_total_vectors * sizeof(int));
        if (!reordered_data || !reordered_indices || !original_indices) {
            if (need_free_h_data) std::free(h_data);
            if (reordered_data) std::free(reordered_data);
            if (reordered_indices) std::free(reordered_indices);
            if (original_indices) std::free(original_indices);
            cudaFree(d_centroids);
            cudaFreeHost(h_init_centroids);
            throw std::bad_alloc();
        }
        
        // Step 4.1: 初始化原始索引数组 [0, 1, 2, ..., n_total_vectors-1]
        for (int i = 0; i < n_total_vectors; ++i) {
            original_indices[i] = i;
        }
        
        // Step 5: 运行IVF K-means（聚类 + 重排数据和索引）
        // 使用两个不同的数组：original_indices 作为输入，reordered_indices 作为输出
        // 注意：h_data_ptr 不会被修改，输出到 reordered_data
        bool success = ivf_kmeans(kmeans_cfg, h_data_ptr, reordered_data, d_centroids,
                                 &cluster_info, use_minibatch, device_id, batch_size, h_objective,
                                 original_indices, reordered_indices);
        if (!success) {
            if (need_free_h_data) std::free(h_data);
            std::free(reordered_data);
            std::free(reordered_indices);
            std::free(original_indices);
            cudaFree(d_centroids);
            cudaFreeHost(h_init_centroids);
            throw std::runtime_error("ivf_kmeans failed");
        }
        
        // 释放临时原始索引数组
        std::free(original_indices);
        cudaDeviceSynchronize();
        
        // Step 6: 拷贝最终的centroids回host
        centroids = (float*)memalign(64, sizeof(float) * (size_t)n_clusters * vector_dim);
        cudaMemcpy(centroids, d_centroids, sizeof(float) * (size_t)n_clusters * vector_dim, cudaMemcpyDeviceToHost);
        
        // 清理临时内存
        if (need_free_h_data) std::free(h_data);
        cudaFree(d_centroids);
        cudaFreeHost(h_init_centroids);
    }    
    /**
     * 释放ClusterDataset占用的内存
     */
    void release() {
        // 释放连续存储的数据
        if (reordered_data) {
            std::free(reordered_data);
            reordered_data = nullptr;
        }
        
        // 释放索引数组
        if (reordered_indices) {
            std::free(reordered_indices);
            reordered_indices = nullptr;
        }
        
        // 释放centroids
        if (centroids) {
            std::free(centroids);
            centroids = nullptr;
        }
        
        // 释放 ClusterInfo
        if (cluster_info.offsets) {
            std::free(cluster_info.offsets);
            cluster_info.offsets = nullptr;
        }
        if (cluster_info.counts) {
            std::free(cluster_info.counts);
            cluster_info.counts = nullptr;
        }
        
        cluster_info.k = 0;
        n_total_vectors = 0;
        vector_dim = 0;
    }
    
    /**
     * 检查ClusterDataset是否已初始化
     */
    bool is_valid() const {
        return reordered_data != nullptr && 
               reordered_indices != nullptr &&
               centroids != nullptr &&
               cluster_info.offsets != nullptr &&
               cluster_info.counts != nullptr &&
               cluster_info.k > 0 &&
               n_total_vectors > 0 &&
               vector_dim > 0;
    }
    
    /**
     * 获取cluster数量
     */
    int get_n_clusters() const {
        return cluster_info.k;
    }
};

#endif // CLUSTER_DATASET_CUH

