/* 必须在任何头文件之前包含limits.h，以便Thrust可以使用CHAR_MIN等宏 */
#ifndef _LIMITS_H_
#define _LIMITS_H_
#endif
#include <limits.h>
#include "../pch.h"
#include "integrate_screen.cuh"

#include "../fusion_cos_topk/fusion_cos_topk.cuh"
#include "../indexed_gemm/indexed_gemm.cuh"
#include "../warpsortfilter/warpsort_utils.cuh"
#include "../warpsortfilter/warpsort_topk.cu"
#include "../cudatimer.h"
#include "../../unit_tests/common/test_utils.cuh"
#include "../l2norm/l2norm.cuh"
#include "../utils.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define ENABLE_CUDA_TIMING 0

using namespace pgvector::warpsort_utils;
using namespace pgvector::warpsort_topk;

/**
 * 流水线版本的数据结构：常驻GPU内存的数据
 * 当向量数据总数小于6G时，整个数据集常驻内存
 */
struct PipelinePersistentData {
    // 向量数据（常驻）
    float* d_cluster_vectors = nullptr;  // 所有cluster的向量数据
    float** d_cluster_vector_ptr = nullptr;  // 每个cluster的起始地址
    float* d_cluster_vector_norm = nullptr;  // 向量L2范数
    
    // Cluster元数据（常驻）
    int* d_probe_vector_offset = nullptr;  // [n_total_clusters + 1]
    int* d_probe_vector_count = nullptr;  // [n_total_clusters]
    
    // 聚类中心数据（常驻）
    float* d_cluster_centers = nullptr;
    float* d_cluster_centers_norm = nullptr;
    
    // 数据维度信息
    int n_total_clusters = 0;
    int n_total_vectors = 0;
    int n_dim = 0;
    
    // 数据大小（用于判断是否需要常驻）
    size_t total_data_size_bytes = 0;
    
    // 是否已初始化
    bool initialized = false;
    
    /**
     * 检查数据是否可以常驻内存（小于6G）
     * 
     * @param n_total_vectors 向量总数
     * @param n_dim 向量维度
     * @param n_total_clusters 聚类总数（可选，如果为0则估算）
     */
    bool can_persist(size_t n_total_vectors, int n_dim, size_t n_total_clusters = 0) {
        // 如果没有提供 n_total_clusters，则估算（假设每个 cluster 有 sqrt(n_total_vectors) 个向量）
        if (n_total_clusters == 0) {
            n_total_clusters = static_cast<size_t>(std::sqrt(static_cast<double>(n_total_vectors)));
            if (n_total_clusters == 0) n_total_clusters = 1;
        }
        
        size_t vector_data_size = n_total_vectors * n_dim * sizeof(float);
        size_t norm_data_size = n_total_vectors * sizeof(float);
        size_t cluster_center_size = n_total_clusters * n_dim * sizeof(float);
        size_t cluster_center_norm_size = n_total_clusters * sizeof(float);
        size_t metadata_size = (n_total_clusters + 1) * sizeof(int) + n_total_clusters * sizeof(int) + n_total_clusters * sizeof(float*);
        total_data_size_bytes = vector_data_size + norm_data_size + cluster_center_size + cluster_center_norm_size + metadata_size;
        return total_data_size_bytes < (6ULL * 1024 * 1024 * 1024);  // 6GB
    }
    
    /**
     * 初始化常驻数据
     */
    void initialize(int* cluster_size,
                    float*** cluster_vectors,
                    float** cluster_center_data,
                    int n_total_clusters,
                    int n_total_vectors,
                    int n_dim,
                    cudaStream_t stream = 0) {
        if (initialized) {
            // 检查维度是否匹配
            if (this->n_total_clusters != n_total_clusters ||
                this->n_total_vectors != n_total_vectors ||
                this->n_dim != n_dim) {
                // 维度不匹配，需要重新初始化
                cleanup();
            } else {
                // 维度匹配，数据已存在，无需重新初始化
                return;
            }
        }
        
        this->n_total_clusters = n_total_clusters;
        this->n_total_vectors = n_total_vectors;
        this->n_dim = n_dim;
        
        // 分配cluster向量指针数组（CPU端）
        d_cluster_vector_ptr = (float**)malloc(n_total_clusters * sizeof(float*));
        
        // 分配GPU内存
        cudaMalloc(&d_cluster_vectors, n_total_vectors * n_dim * sizeof(float));
        cudaMalloc(&d_probe_vector_offset, (n_total_clusters + 1) * sizeof(int));
        cudaMalloc(&d_probe_vector_count, n_total_clusters * sizeof(int));
        cudaMalloc(&d_cluster_centers, n_total_clusters * n_dim * sizeof(float));
        cudaMalloc(&d_cluster_vector_norm, n_total_vectors * sizeof(float));
        cudaMalloc(&d_cluster_centers_norm, n_total_clusters * sizeof(float));
        CHECK_CUDA_ERRORS;
        
        // 复制cluster_size到GPU并计算offset
        cudaMemcpyAsync(d_probe_vector_count, cluster_size, 
                       n_total_clusters * sizeof(int), cudaMemcpyHostToDevice, stream);
        compute_prefix_sum(d_probe_vector_count, d_probe_vector_offset, n_total_clusters, stream);
        
        // 计算每个cluster的起始位置（在CPU端计算，避免同步）
        // 使用cluster_size在CPU端计算offset，避免从GPU读取
        int* probe_vector_offset_host = (int*)malloc((n_total_clusters + 1) * sizeof(int));
        probe_vector_offset_host[0] = 0;
        for (int i = 0; i < n_total_clusters; ++i) {
            probe_vector_offset_host[i + 1] = probe_vector_offset_host[i] + cluster_size[i];
        }
        
        // 复制cluster向量数据到GPU（使用异步传输）
        // 所有传输都在同一个stream中，CUDA会保证顺序执行
        for (int i = 0; i < n_total_clusters; ++i) {
            float* cluster_start = d_cluster_vectors + probe_vector_offset_host[i] * n_dim;
            cudaMemcpyAsync(cluster_start, cluster_vectors[i][0], 
                            cluster_size[i] * n_dim * sizeof(float), 
                            cudaMemcpyHostToDevice, stream);
            d_cluster_vector_ptr[i] = cluster_start;  // 存储GPU设备指针
        }
        
        // 复制cluster中心数据
        cudaMemcpyAsync(d_cluster_centers, cluster_center_data[0], 
                       n_total_clusters * n_dim * sizeof(float), 
                       cudaMemcpyHostToDevice, stream);
        
        cudaStreamSynchronize(stream);
    
        // 计算L2范数（在stream中异步执行）
        // 注意：由于所有操作都在同一个stream中，CUDA会保证顺序执行
        // 即 L2 范数计算会在数据复制完成后执行
        compute_l2_norm_gpu(d_cluster_vectors, d_cluster_vector_norm, n_total_vectors, n_dim, L2NORM_AUTO, stream);
        compute_l2_norm_gpu(d_cluster_centers, d_cluster_centers_norm, n_total_clusters, n_dim, L2NORM_AUTO, stream);
        
        free(probe_vector_offset_host);
        
        // 注意：initialized 标志在异步操作完成后才设置
        // 如果 stream 是默认 stream (0)，则立即设置
        // 如果 stream 是异步 stream，需要调用者同步后再设置
        if (stream == 0 || stream == nullptr) {
            // 默认 stream，操作是同步的，可以立即设置
            initialized = true;
        }
        // 否则，initialized 将在调用者同步 stream 后设置
    }
    
    /**
     * 标记初始化完成（在 stream 同步后调用）
     */
    void mark_initialized() {
        initialized = true;
    }
    
    /**
     * 清理常驻数据
     */
    void cleanup() {
        if (d_cluster_vectors != nullptr) {
            cudaFree(d_cluster_vectors);
            d_cluster_vectors = nullptr;
        }
        if (d_cluster_vector_ptr != nullptr) {
            free(d_cluster_vector_ptr);
            d_cluster_vector_ptr = nullptr;
        }
        if (d_cluster_vector_norm != nullptr) {
            cudaFree(d_cluster_vector_norm);
            d_cluster_vector_norm = nullptr;
        }
        if (d_probe_vector_offset != nullptr) {
            cudaFree(d_probe_vector_offset);
            d_probe_vector_offset = nullptr;
        }
        if (d_probe_vector_count != nullptr) {
            cudaFree(d_probe_vector_count);
            d_probe_vector_count = nullptr;
        }
        if (d_cluster_centers != nullptr) {
            cudaFree(d_cluster_centers);
            d_cluster_centers = nullptr;
        }
        if (d_cluster_centers_norm != nullptr) {
            cudaFree(d_cluster_centers_norm);
            d_cluster_centers_norm = nullptr;
        }
        // 重置所有状态
        initialized = false;
        n_total_clusters = 0;
        n_total_vectors = 0;
        n_dim = 0;
        total_data_size_bytes = 0;
    }
};

// 全局常驻数据（简单版本：单实例）
static PipelinePersistentData g_persistent_data;

/**
 * 初始化常驻数据集（可在单元测试的计时之外调用）
 */
bool initialize_persistent_data(
    int* cluster_size,
    float*** cluster_vectors,
    float** cluster_center_data,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim)
{
    // 检查是否可以常驻内存
    if (!g_persistent_data.can_persist(n_total_vectors, n_dim, n_total_clusters)) {
        return false;
    }
    
    // 创建临时stream用于初始化
    cudaStream_t init_stream;
    cudaStreamCreate(&init_stream);
    
    // 初始化常驻数据（同步执行，确保初始化完成）
    g_persistent_data.initialize(cluster_size, cluster_vectors, cluster_center_data,
                                 n_total_clusters, n_total_vectors, n_dim, init_stream);
    
    // 标记初始化完成
    g_persistent_data.mark_initialized();
    
    cudaStreamDestroy(init_stream);
    CHECK_CUDA_ERRORS;
    
    return true;
}

/**
 * 清理常驻数据集
 */
void cleanup_persistent_data()
{
    g_persistent_data.cleanup();
}

/**
 * 流水线版本的批量查询接口
 * 
 * 流水线分为两部分：
 * 1. 向量数据准备：上传query数据、计算query norm等
 * 2. 核函数计算：粗筛、精筛等计算
 * 
 * 当向量数据总数小于6G时，整个数据集常驻GPU内存
 */
void batch_search_pipeline(float** query_batch,
                           int* cluster_size,
                           float*** cluster_vectors,
                           float** cluster_center_data,
                           int* initial_indices,
                           float** topk_dist,
                           int** topk_index,
                           int* n_isnull,

                           int n_query,
                           int n_dim,
                           int n_total_clusters,
                           int n_total_vectors,
                           int n_probes,
                           int k) {

    if (n_query <= 0 || n_dim <= 0 || n_total_clusters <= 0 || k <= 0) {
        printf("[ERROR] Invalid parameters: n_query=%d, n_dim=%d, n_total_clusters=%d, k=%d\n",
               n_query, n_dim, n_total_clusters, k);
        throw std::invalid_argument("invalid batch_search_pipeline configuration");
    }
    if (!cluster_size || !cluster_vectors) {
        throw std::invalid_argument("cluster metadata is null");
    }

    if (!cluster_center_data) {
        throw std::invalid_argument("cluster_center_data must not be null for coarse search");
    }
    if (n_probes <= 0 || n_probes > n_total_clusters) {
        throw std::invalid_argument("invalid n_probes");
    }

    // 创建CUDA streams用于流水线
    cudaStream_t data_stream, compute_stream;
    cudaStreamCreate(&data_stream);
    cudaStreamCreate(&compute_stream);
    
    // 创建事件用于同步
    cudaEvent_t data_ready_event;
    cudaEventCreate(&data_ready_event);
    
    // 检查是否可以常驻内存
    bool use_persistent = g_persistent_data.can_persist(n_total_vectors, n_dim, n_total_clusters);
    
    // 临时数据（每次查询都需要）
    float* d_queries = nullptr;
    float* d_query_norm = nullptr;
    int* d_topk_index = nullptr;
    float* d_topk_dist = nullptr;
    int* d_top_nprobe_index = nullptr;
    float* d_top_nprobe_dist = nullptr;
    float* d_inner_product = nullptr;

    dim3 queryDim(n_query);
    dim3 dataDim(n_total_clusters);
    dim3 vectorDim(n_dim);
    dim3 probeDim(n_probes);

    // ==================================================================
    // 流水线阶段1：向量数据准备（在data_stream中异步执行）
    // ==================================================================
    {
        CUDATimer timer("Pipeline Stage 1: Data Preparation");
        
        // 分配query相关内存
        cudaMalloc(&d_queries, n_query * n_dim * sizeof(float));
        cudaMalloc(&d_query_norm, n_query * sizeof(float));
        cudaMalloc(&d_topk_dist, n_query * k * sizeof(float));
        cudaMalloc(&d_topk_index, n_query * k * sizeof(int));
        cudaMalloc(&d_top_nprobe_index, n_query * n_probes * sizeof(int));
        CHECK_CUDA_ERRORS;
        
        // 初始化query norm为0（虽然compute_l2_norm_gpu会覆盖，但为了安全起见）
        cudaMemsetAsync(d_query_norm, 0, n_query * sizeof(float), data_stream);
        
        // 如果可以使用常驻数据，检查是否需要初始化或重新初始化
        // 注意：如果已在计时外初始化，但维度可能已改变，需要检查并重新初始化
        if (use_persistent) {
            // 检查维度是否匹配（即使已初始化，如果维度不匹配也需要重新初始化）
            bool need_reinit = !g_persistent_data.initialized ||
                              g_persistent_data.n_total_clusters != n_total_clusters ||
                              g_persistent_data.n_total_vectors != n_total_vectors ||
                              g_persistent_data.n_dim != n_dim;
            
            if (need_reinit) {
                // 在data_stream中异步初始化或重新初始化常驻数据
                // initialize() 内部会检查维度，如果不匹配会自动 cleanup() 后重新初始化
                g_persistent_data.initialize(cluster_size, cluster_vectors, cluster_center_data,
                                            n_total_clusters, n_total_vectors, n_dim, data_stream);
                // 注意：如果重新初始化，需要等待初始化完成后再继续
                // 因为后续的 query 数据传输和 norm 计算依赖于正确的 n_dim
            }
        } else if (!use_persistent) {
            // TODO: 实现非常驻模式的数据上传
            // 当前简单版本只支持常驻模式
            throw std::runtime_error("Non-persistent mode not implemented yet");
        }
        
        // 异步复制query数据到GPU（与常驻数据初始化并行执行）
        // 注意：query_batch[0] 指向所有query的连续内存块的起始地址
        // query_batch 是通过 malloc_vector_list 分配的，数据是连续存储的
        // 因此可以使用 query_batch[0] 一次性复制所有 query 数据
        // 
        // 重要：确保 query_batch 是用当前的 n_dim 生成的！
        // 如果 n_dim 改变了，调用者必须重新生成 query_batch
        // 
        // 内存布局：query_batch[0] 指向连续内存 [query0, query1, ..., queryN]
        // 每个 query 占用 n_dim * sizeof(float) 字节
        // query_batch[i] = query_batch[0] + i * n_dim * sizeof(float)
        // 
        // 因此，d_queries 的内存布局应该是：
        // d_queries[0 ... n_dim-1] = query0
        // d_queries[n_dim ... 2*n_dim-1] = query1
        // ...
        // d_queries[i*n_dim ... (i+1)*n_dim-1] = query_i
        cudaMemcpyAsync(d_queries, query_batch[0], 
                       n_query * n_dim * sizeof(float), 
                       cudaMemcpyHostToDevice, data_stream);
        
        // 异步计算query norm（在data_stream中）
        // 注意：compute_l2_norm_gpu 会为每个 query 计算 L2 范数
        // d_query_norm[i] = ||d_queries[i * n_dim : (i+1) * n_dim]||_2
        // 这里必须使用正确的 n_dim，否则会计算错误的 norm
        // 
        // 如果 n_dim 改变了，但 query_batch 是用旧的 n_dim 生成的，那么：
        // - 如果新的 n_dim > 旧的 n_dim：会读取越界数据（但第一个 query 可能刚好在边界内）
        // - 如果新的 n_dim < 旧的 n_dim：会只复制部分数据，导致后续 query 的数据错位
        compute_l2_norm_gpu(d_queries, d_query_norm, n_query, n_dim, L2NORM_AUTO, data_stream);
        
        cudaStreamSynchronize(data_stream);
        CHECK_CUDA_ERRORS;
        
        // 记录数据准备完成事件（确保所有数据都已准备好）
        // 此时data_stream中的所有操作（query传输、query norm计算、常驻数据初始化）都应该已完成
        cudaEventRecord(data_ready_event, data_stream);
    }

    // ==================================================================
    // 流水线阶段2：核函数计算（在compute_stream中执行）
    // ==================================================================
    
    // 等待数据准备完成（使用事件同步，而不是stream同步）
    cudaStreamWaitEvent(compute_stream, data_ready_event, 0);
    
    // Step 1: 粗筛
    {
        CUDATimer timer("Pipeline Stage 2: Coarse Search");
        
        float alpha = 1.0f; 
        float beta = 0.0f;
        cublasHandle_t handle;
        int* d_index = nullptr;
        bool need_free_index = false;
    
        cudaMalloc(&d_inner_product, n_query * n_total_clusters * sizeof(float));
        cudaMalloc(&d_top_nprobe_dist, n_query * n_probes * sizeof(float));
            cublasCreate(&handle);
        CHECK_CUDA_ERRORS;
    
        // 使用外部传入的索引，如果没有传入则内部生成顺序索引
        if (initial_indices != nullptr) {
            // 复制外部传入的索引到GPU
            cudaMalloc(&d_index, n_query * n_total_clusters * sizeof(int));
            cudaMemcpyAsync(d_index, initial_indices, 
                           n_query * n_total_clusters * sizeof(int),
                           cudaMemcpyHostToDevice, compute_stream);
            need_free_index = true;
        } else {
            // 内部生成顺序索引
            cudaMalloc(&d_index, n_query * n_total_clusters * sizeof(int));
            need_free_index = true;
            dim3 block_dim((n_total_clusters < 256) ? n_total_clusters : 256);
        generate_sequence_indices_kernel<<<queryDim, block_dim, 0, compute_stream>>>(
                d_index, n_query, n_total_clusters);
        }
        
        // 初始化距离数组（使用fill kernel替代thrust::fill）
        dim3 fill_block(256);
        int fill_grid_size = (n_query * n_probes + fill_block.x - 1) / fill_block.x;
        dim3 fill_grid(fill_grid_size);
        fill_kernel<<<fill_grid, fill_block, 0, compute_stream>>>(
            d_top_nprobe_dist,
            FLT_MAX,
            n_query * n_probes
        );
        
        // 矩阵乘法（使用compute_stream）
        cublasSetStream(handle, compute_stream);
            cublasSgemm(handle, 
                CUBLAS_OP_T, CUBLAS_OP_N, 
                n_total_clusters, n_query, n_dim,                   
                &alpha, 
            g_persistent_data.d_cluster_centers, n_dim,            
                d_queries, n_dim,               
                &beta, 
                d_inner_product, n_total_clusters
            );    
            
        // 余弦距离 + topk（使用compute_stream）
            pgvector::fusion_cos_topk_warpsort::fusion_cos_topk_warpsort<float, int>(
            d_query_norm, g_persistent_data.d_cluster_centers_norm, d_inner_product, d_index,
            n_query, n_total_clusters, n_probes,
                d_top_nprobe_dist, d_top_nprobe_index,
            true /* select min */,
            compute_stream  // 使用compute_stream，确保与数据准备同步
            );

        cudaStreamSynchronize(compute_stream);
            CHECK_CUDA_ERRORS;
            
        // 清理临时内存
            cublasDestroy(handle);
        // 注意：d_cluster_centers 来自常驻数据，不应释放
            cudaFree(d_inner_product);
            if (need_free_index && d_index != nullptr) {
            cudaFree(d_index);
            }
            cudaFree(d_top_nprobe_dist);
        CHECK_CUDA_ERRORS;
    }

    // Step 2: 构建entry数据
    int* d_cluster_query_offset = nullptr;
    int* d_cluster_query_data = nullptr;
    int* d_cluster_query_probe_indices = nullptr;
    int* d_entry_cluster_id = nullptr;
    int* d_entry_query_start = nullptr;
    int* d_entry_query_count = nullptr;
    int* d_entry_queries = nullptr;
    int* d_entry_probe_indices = nullptr;
    int n_entry = 0;
    constexpr int kQueriesPerBlock = 8;

    {
        CUDATimer timer("Pipeline Stage 2: Build Entry Data");
        
        // 统计每个cluster的query数量
        int* d_cluster_query_count = nullptr;
        cudaMalloc(&d_cluster_query_count, n_total_clusters * sizeof(int));
        cudaMemsetAsync(d_cluster_query_count, 0, n_total_clusters * sizeof(int), compute_stream);
        
        count_cluster_queries_kernel<<<queryDim, probeDim, 0, compute_stream>>>(
            d_top_nprobe_index,
            d_cluster_query_count,
            n_query,
            n_probes,
            n_total_clusters
        );
        
        // 计算offset
        cudaMalloc(&d_cluster_query_offset, (n_total_clusters + 1) * sizeof(int));
        compute_prefix_sum(d_cluster_query_count, d_cluster_query_offset, n_total_clusters, compute_stream);
        
        int total_entries = 0;
        cudaMemcpyAsync(&total_entries, d_cluster_query_offset + n_total_clusters, 
                       sizeof(int), cudaMemcpyDeviceToHost, compute_stream);
        cudaStreamSynchronize(compute_stream);
        CHECK_CUDA_ERRORS;
        
        // 构建cluster-query映射
        int* d_cluster_write_pos = nullptr;
        cudaMalloc(&d_cluster_write_pos, n_total_clusters * sizeof(int));
        cudaMemcpyAsync(d_cluster_write_pos, d_cluster_query_offset, 
                       n_total_clusters * sizeof(int), cudaMemcpyDeviceToDevice, compute_stream);
        
        cudaMalloc(&d_cluster_query_data, total_entries * sizeof(int));
        cudaMalloc(&d_cluster_query_probe_indices, total_entries * sizeof(int));
        
        build_cluster_query_mapping_kernel<<<queryDim, probeDim, 0, compute_stream>>>(
            d_top_nprobe_index,
            d_cluster_query_offset,
            d_cluster_query_data,
            d_cluster_query_probe_indices,
            d_cluster_write_pos,
            n_query,
            n_probes,
            n_total_clusters
        );
        
        // 计算entry数量
        int* d_entry_count_per_cluster = nullptr;
        cudaMalloc(&d_entry_count_per_cluster, n_total_clusters * sizeof(int));
        
        dim3 clusterDim(n_total_clusters);
        dim3 blockDim_entry(1);
        count_entries_per_cluster_kernel<<<clusterDim, blockDim_entry, 0, compute_stream>>>(
            d_cluster_query_offset,
            d_entry_count_per_cluster,
            n_total_clusters,
            kQueriesPerBlock
        );
        
        // 计算entry offset
        int* d_entry_offset = nullptr;
        cudaMalloc(&d_entry_offset, (n_total_clusters + 1) * sizeof(int));
        compute_prefix_sum(d_entry_count_per_cluster, d_entry_offset, n_total_clusters, compute_stream);
        
        cudaMemcpyAsync(&n_entry, d_entry_offset + n_total_clusters, 
                       sizeof(int), cudaMemcpyDeviceToHost, compute_stream);
        cudaStreamSynchronize(compute_stream);
        CHECK_CUDA_ERRORS;
        
        // 构建entry数据
        int* d_entry_query_offset = nullptr;
        cudaMalloc(&d_entry_query_offset, (n_total_clusters + 1) * sizeof(int));
        cudaMemcpyAsync(d_entry_query_offset, d_cluster_query_offset, 
                       (n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToDevice, compute_stream);
        
        if (n_entry > 0) {
            cudaMalloc(&d_entry_cluster_id, n_entry * sizeof(int));
            cudaMalloc(&d_entry_query_start, n_entry * sizeof(int));
            cudaMalloc(&d_entry_query_count, n_entry * sizeof(int));
            cudaMalloc(&d_entry_queries, total_entries * sizeof(int));
            cudaMalloc(&d_entry_probe_indices, total_entries * sizeof(int));
            
            build_entry_data_kernel<<<clusterDim, blockDim_entry, 0, compute_stream>>>(
                d_cluster_query_offset,
                d_cluster_query_data,
                d_cluster_query_probe_indices,
                d_entry_offset,
                d_entry_query_offset,
                d_entry_cluster_id,
                d_entry_query_start,
                d_entry_query_count,
                d_entry_queries,
                d_entry_probe_indices,
                n_total_clusters,
                kQueriesPerBlock
            );
        }
        
        cudaStreamSynchronize(compute_stream);
        CHECK_CUDA_ERRORS;
        
        // 清理临时内存
        cudaFree(d_cluster_query_count);
        cudaFree(d_cluster_write_pos);
        cudaFree(d_entry_count_per_cluster);
        cudaFree(d_entry_offset);
        cudaFree(d_entry_query_offset);
        cudaFree(d_top_nprobe_index);
    }

    // 确保数据准备完成（再次等待，确保query数据已准备好）
    // 虽然粗筛阶段已经等待了，但为了安全起见，精筛阶段也再次等待
    cudaStreamWaitEvent(compute_stream, data_ready_event, 0);

    // Step 3: 精筛
    {
        CUDATimer timer("Pipeline Stage 2: Fine Search");

        int capacity = 32;
        while (capacity < k) capacity <<= 1;
        capacity = std::min(capacity, kMaxCapacity);
        
        float* d_topk_dist_candidate = nullptr;
        int* d_topk_index_candidate = nullptr;
            cudaMalloc(&d_topk_dist_candidate, n_query * n_probes * k * sizeof(float));
            cudaMalloc(&d_topk_index_candidate, n_query * n_probes * k * sizeof(int));
        
            dim3 init_block(512);
            int init_grid_size = (n_query * n_probes * k + init_block.x - 1) / init_block.x;
            dim3 init_grid(init_grid_size);
        init_invalid_values_kernel<<<init_grid, init_block, 0, compute_stream>>>(
                d_topk_dist_candidate,
                d_topk_index_candidate,
                n_query * n_probes * k
            );
        
        dim3 block(kQueriesPerBlock * 32);
        if (n_entry > 0) {
                if (capacity <= 32) {
                    launch_indexed_inner_product_with_topk_kernel_v5_entry_based<64, true, kQueriesPerBlock>(
                    block, n_dim, d_queries,
                    g_persistent_data.d_cluster_vectors,
                    g_persistent_data.d_probe_vector_offset,
                    g_persistent_data.d_probe_vector_count,
                    d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
                    d_entry_queries, d_entry_probe_indices,
                    d_query_norm, g_persistent_data.d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    d_topk_dist_candidate, d_topk_index_candidate, compute_stream
                    );
                } else if (capacity <= 64) {
                    launch_indexed_inner_product_with_topk_kernel_v5_entry_based<128, true, kQueriesPerBlock>(
                    block, n_dim, d_queries,
                    g_persistent_data.d_cluster_vectors,
                    g_persistent_data.d_probe_vector_offset,
                    g_persistent_data.d_probe_vector_count,
                    d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
                    d_entry_queries, d_entry_probe_indices,
                    d_query_norm, g_persistent_data.d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    d_topk_dist_candidate, d_topk_index_candidate, compute_stream
                    );
                } else {
                    launch_indexed_inner_product_with_topk_kernel_v5_entry_based<256, true, kQueriesPerBlock>(
                    block, n_dim, d_queries,
                    g_persistent_data.d_cluster_vectors,
                    g_persistent_data.d_probe_vector_offset,
                    g_persistent_data.d_probe_vector_count,
                    d_entry_cluster_id, d_entry_query_start, d_entry_query_count,
                    d_entry_queries, d_entry_probe_indices,
                    d_query_norm, g_persistent_data.d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    d_topk_dist_candidate, d_topk_index_candidate, compute_stream
                );
            }
        }
        
        // 等待精筛 kernel 完成，确保候选数据已准备好
        cudaStreamSynchronize(compute_stream);
        CHECK_CUDA_ERRORS;
        
        // 规约：从候选中选择top-k
        // 注意：d_topk_dist_candidate 和 d_topk_index_candidate 的格式是 [n_query][n_probes][k]
        // select_k 期望的格式是 [n_query][n_probes * k]，两者是匹配的（展平后）
        // select_k 输出的是候选位置索引（0 到 n_probes * k - 1），需要映射回原始向量索引
            select_k<float, int>(
                d_topk_dist_candidate, n_query, n_probes * k, k,
            d_topk_dist, d_topk_index, true, compute_stream
            );
        
        // 等待 select_k 完成
        cudaStreamSynchronize(compute_stream);
            CHECK_CUDA_ERRORS;
            
        // 验证：检查 select_k 输出的索引是否在有效范围内
        // 临时调试：将 d_topk_index 复制到 CPU 检查
        #ifdef DEBUG_MAP_INDICES
        int* h_topk_index_debug = (int*)malloc(n_query * k * sizeof(int));
        cudaMemcpy(h_topk_index_debug, d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost);
        int max_candidates = n_probes * k;
        for (int i = 0; i < n_query * k; ++i) {
            if (h_topk_index_debug[i] < 0 || h_topk_index_debug[i] >= max_candidates) {
                printf("[DEBUG] Invalid candidate_pos at idx=%d: %d (max=%d)\n", 
                       i, h_topk_index_debug[i], max_candidates);
            }
        }
        free(h_topk_index_debug);
        #endif
        
        // 映射候选索引回原始向量索引
        // d_topk_index 现在包含候选位置索引，需要映射回原始向量索引
        // 注意：d_topk_index_candidate 的格式是 [n_query][n_probes][k]，展平后是 [n_query][n_probes * k]
            dim3 map_block(256);
            dim3 map_grid((n_query * k + map_block.x - 1) / map_block.x);
        map_candidate_indices_kernel<<<map_grid, map_block, 0, compute_stream>>>(
            d_topk_index_candidate,  // 原始索引数组 [n_query][n_probes * k]（展平格式）
            d_topk_index,            // 输入：候选位置索引 [n_query][k]，输出：原始向量索引
                n_query,
                n_probes,
                k
            );
        
        cudaStreamSynchronize(compute_stream);
            CHECK_CUDA_ERRORS;
            
            cudaFree(d_topk_dist_candidate);
            cudaFree(d_topk_index_candidate);
        }
        
    // 复制结果回CPU
    cudaMemcpyAsync(topk_dist[0], d_topk_dist, 
                   n_query * k * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
    cudaMemcpyAsync(topk_index[0], d_topk_index, 
                   n_query * k * sizeof(int), cudaMemcpyDeviceToHost, compute_stream);
    cudaStreamSynchronize(compute_stream);
        CHECK_CUDA_ERRORS;
    
    // 清理临时内存
    cudaFree(d_queries);
    cudaFree(d_query_norm);
    cudaFree(d_topk_dist);
    cudaFree(d_topk_index);
    cudaFree(d_cluster_query_offset);
    cudaFree(d_cluster_query_data);
    cudaFree(d_cluster_query_probe_indices);
    if (d_entry_cluster_id != nullptr) cudaFree(d_entry_cluster_id);
    if (d_entry_query_start != nullptr) cudaFree(d_entry_query_start);
    if (d_entry_query_count != nullptr) cudaFree(d_entry_query_count);
    if (d_entry_queries != nullptr) cudaFree(d_entry_queries);
    if (d_entry_probe_indices != nullptr) cudaFree(d_entry_probe_indices);
    
    // 销毁streams和事件
    cudaEventDestroy(data_ready_event);
    cudaStreamDestroy(data_stream);
    cudaStreamDestroy(compute_stream);
    
    CHECK_CUDA_ERRORS;
}

void run_integrate_pipeline() {
    // TODO: 后续补充粗筛 + 精筛整体调度
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
}
