/* ivf_search_separated.cu */

#ifndef _LIMITS_H_
#define _LIMITS_H_
#endif
#include <limits.h>
#include "../pch.h"

// 包含原始所有需要的头文件
#include "ivf_search.cuh"
#include "../fusion_dist_topk/fusion_dist_topk.cuh"
#include "../indexed_gemm/indexed_gemm.cuh"
#include "../warpsortfilter/warpsort_utils.cuh"
#include "../warpsortfilter/warpsort_topk.cu"
#include "../l2norm/l2norm.cuh"
#include "../../unit_tests/common/test_utils.cuh"
#include "../utils.cuh"
#include <vector>
#include <algorithm>
#include <cfloat>

using namespace pgvector::warpsort_utils;
using namespace pgvector::warpsort_topk;

// ---------------------------------------------------------
// 数据结构定义（与 ivfscanbatch.h 中的定义保持一致）
// 注意：由于 CUDA 文件无法包含 PostgreSQL 头文件，这里直接定义结构体
// ---------------------------------------------------------

/**
 * 索引上下文：管理常驻显存的数据 (Dataset & Clusters)
 * 与 ivfscanbatch.h 中的 IVFIndexContext 保持一致
 */
struct IVFIndexContext {
    float* d_cluster_vectors;
    float* d_cluster_vector_norm;
    int* d_probe_vector_offset;
    int* d_probe_vector_count;
    float* d_cluster_centers;
    float* d_cluster_centers_norm;
    int n_total_clusters;
    int n_total_vectors;
    int n_dim;
    bool is_initialized;
};

/**
 * 查询批次上下文：管理每个Batch的临时资源和Stream
 * 与 ivfscanbatch.h 中的 IVFQueryBatchContext 保持一致
 */
struct IVFQueryBatchContext {
    void* stream;                    /* cudaStream_t */
    void* data_ready_event;          /* cudaEvent_t */
    void* compute_done_event;         /* cudaEvent_t */
    float* d_queries;
    float* d_query_norm;
    float* d_inner_product;
    float* d_top_nprobe_dist;
    int* d_top_nprobe_index;
    int* d_index_seq;
    int* d_cluster_query_count;
    int* d_cluster_query_offset;
    int* d_cluster_query_data;
    int* d_cluster_query_probe_indices;
    int* d_cluster_write_pos;
    int* d_entry_count_per_cluster;
    int* d_entry_offset;
    int* d_entry_query_offset;
    int* d_entry_cluster_id;
    int* d_entry_query_start;
    int* d_entry_query_count;
    int* d_entry_queries;
    int* d_entry_probe_indices;
    float* d_topk_dist_candidate;
    int* d_topk_index_candidate;
    float* d_topk_dist;
    int* d_topk_index;
    int max_n_query;
    int n_dim;
    int max_n_probes;
    int max_k;
    int n_total_clusters;
};

// 内部辅助函数：将 void* 转换为 CUDA 类型
static inline cudaStream_t get_stream(IVFQueryBatchContext* ctx) {
    return static_cast<cudaStream_t>(ctx->stream);
}

static inline cudaEvent_t get_data_ready_event(IVFQueryBatchContext* ctx) {
    return static_cast<cudaEvent_t>(ctx->data_ready_event);
}

static inline cudaEvent_t get_compute_done_event(IVFQueryBatchContext* ctx) {
    return static_cast<cudaEvent_t>(ctx->compute_done_event);
}

// ---------------------------------------------------------
// 1. Index 管理 (数据常驻)
// ---------------------------------------------------------

void* ivf_create_index_context() {
    return new IVFIndexContext{0};
}

void ivf_destroy_index_context(void* ctx_ptr) {
    if (!ctx_ptr) return;
    IVFIndexContext* ctx = (IVFIndexContext*)ctx_ptr;
    
    // 释放所有由上下文分配的内存
    if (ctx->d_cluster_vectors) cudaFree(ctx->d_cluster_vectors);
    if (ctx->d_cluster_vector_norm) cudaFree(ctx->d_cluster_vector_norm);
    if (ctx->d_probe_vector_offset) cudaFree(ctx->d_probe_vector_offset);
    if (ctx->d_probe_vector_count) cudaFree(ctx->d_probe_vector_count);
    if (ctx->d_cluster_centers) cudaFree(ctx->d_cluster_centers);
    if (ctx->d_cluster_centers_norm) cudaFree(ctx->d_cluster_centers_norm);
    
    delete ctx;
}

// 加载数据集到 GPU (相当于 Stage 0)
int ivf_load_dataset(
    void* idx_ctx_ptr,
    int* d_cluster_size,
    float* d_cluster_vectors,
    float* d_cluster_centers,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
) {
    IVFIndexContext* ctx = (IVFIndexContext*)idx_ctx_ptr;
    if (ctx->is_initialized) return 0; // 防止重复初始化
    
    ctx->n_total_clusters = n_total_clusters;
    ctx->n_total_vectors = n_total_vectors;
    ctx->n_dim = n_dim;
    
    // 1. 分配内存
    cudaMalloc(&ctx->d_probe_vector_offset, (n_total_clusters + 1) * sizeof(int));
    cudaMalloc(&ctx->d_cluster_vector_norm, n_total_vectors * sizeof(float));
    cudaMalloc(&ctx->d_cluster_centers_norm, n_total_clusters * sizeof(float));
    
    // 2. 保存传入的 device 指针
    ctx->d_cluster_vectors = d_cluster_vectors;
    ctx->d_probe_vector_count = d_cluster_size;
    ctx->d_cluster_centers = d_cluster_centers;
    
    // 3. 计算Offset（使用默认流，因为是初始化阶段）
    compute_prefix_sum(ctx->d_probe_vector_count, ctx->d_probe_vector_offset, n_total_clusters, 0);
    
    // 4. 计算 Norm
    compute_l2_norm_gpu(ctx->d_cluster_vectors, ctx->d_cluster_vector_norm, n_total_vectors, n_dim, L2NORM_AUTO, 0);
    compute_l2_norm_gpu(ctx->d_cluster_centers, ctx->d_cluster_centers_norm, n_total_clusters, n_dim, L2NORM_AUTO, 0);
    
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    
    ctx->is_initialized = true;
    return 1;
}

// ---------------------------------------------------------
// 流式上传接口（在 Build 过程中使用）
// ---------------------------------------------------------

/**
 * 初始化流式上传：分配 GPU 显存空间（不填充数据）
 */
void ivf_init_streaming_upload(
    void* idx_ctx_ptr,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
) {
    IVFIndexContext* ctx = (IVFIndexContext*)idx_ctx_ptr;
    
    ctx->n_total_clusters = n_total_clusters;
    ctx->n_total_vectors = n_total_vectors;
    ctx->n_dim = n_dim;
    
    // 一次性分配好全部显存
    cudaMalloc(&ctx->d_cluster_vectors, (size_t)n_total_vectors * n_dim * sizeof(float));
    cudaMalloc(&ctx->d_probe_vector_offset, (n_total_clusters + 1) * sizeof(int));
    cudaMalloc(&ctx->d_probe_vector_count, n_total_clusters * sizeof(int));
    
    // 初始化 offset 数组的第一个元素为 0
    int zero = 0;
    cudaMemcpy(ctx->d_probe_vector_offset, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    // 初始化状态
    ctx->is_initialized = false;
    CHECK_CUDA_ERRORS;
}

/**
 * 上传单个 Cluster 的数据 (Append Mode)
 * 
 * @param idx_ctx_ptr 索引上下文指针
 * @param cluster_id 聚类 ID
 * @param host_vector_data CPU 端向量数据（连续存储，大小 = count * dim）
 * @param count 该 cluster 的向量数量
 * @param start_offset_idx 该 cluster 在全局向量数组中的起始索引
 */
void ivf_append_cluster_data(
    void* idx_ctx_ptr,
    int cluster_id,
    float* host_vector_data,
    int count,
    int start_offset_idx
) {
    IVFIndexContext* ctx = (IVFIndexContext*)idx_ctx_ptr;
    
    if (count <= 0 || host_vector_data == nullptr) {
        // 空 cluster，只记录 count 和 offset
        int zero = 0;
        cudaMemcpy(ctx->d_probe_vector_count + cluster_id, &zero, sizeof(int), cudaMemcpyHostToDevice);
        // 【修复】offset[cluster_id] 是 cluster_id 的起始位置，不是 offset[cluster_id + 1]
        // 注意：cluster_id = 0 时，offset[0] 已在初始化时设为 0，这里可以安全覆盖
        cudaMemcpy(ctx->d_probe_vector_offset + cluster_id, &start_offset_idx, sizeof(int), cudaMemcpyHostToDevice);
        return;
    }
    
    size_t dim_size = sizeof(float) * ctx->n_dim;
    
    // 1. 计算 GPU 目标地址
    float* d_dest = ctx->d_cluster_vectors + (size_t)start_offset_idx * ctx->n_dim;
    
    // 2. 异步拷贝数据（使用默认流）
    cudaMemcpy(d_dest, host_vector_data, count * dim_size, cudaMemcpyHostToDevice);
    
    // 3. 记录元数据 (Count 和 Offset) 到 GPU
    cudaMemcpy(ctx->d_probe_vector_count + cluster_id, &count, sizeof(int), cudaMemcpyHostToDevice);
    // 【修复】offset[cluster_id] 是 cluster_id 的起始位置，不是 offset[cluster_id + 1]
    // 注意：cluster_id = 0 时，offset[0] 已在初始化时设为 0，这里可以安全覆盖
    cudaMemcpy(ctx->d_probe_vector_offset + cluster_id, &start_offset_idx, sizeof(int), cudaMemcpyHostToDevice);
    
    CHECK_CUDA_ERRORS;
}

/**
 * 完成流式上传：上传聚类中心，补全最后一个 Offset，计算 Norm
 * 
 * @param idx_ctx_ptr 索引上下文指针
 * @param center_data_flat 展平的聚类中心数据（连续存储，大小 = n_total_clusters * dim）
 * @param total_vectors_check 总向量数（用于校验）
 */
void ivf_finalize_streaming_upload(
    void* idx_ctx_ptr,
    float* center_data_flat,
    int total_vectors_check
) {
    IVFIndexContext* ctx = (IVFIndexContext*)idx_ctx_ptr;
    
    // 补全最后一个 offset (total count)
    cudaMemcpy(ctx->d_probe_vector_offset + ctx->n_total_clusters, &total_vectors_check, sizeof(int), cudaMemcpyHostToDevice);
    
    // 上传聚类中心
    cudaMalloc(&ctx->d_cluster_centers, ctx->n_total_clusters * ctx->n_dim * sizeof(float));
    cudaMemcpy(ctx->d_cluster_centers, center_data_flat, ctx->n_total_clusters * ctx->n_dim * sizeof(float), cudaMemcpyHostToDevice);
    
    // 分配 Norm 内存
    cudaMalloc(&ctx->d_cluster_vector_norm, ctx->n_total_vectors * sizeof(float));
    cudaMalloc(&ctx->d_cluster_centers_norm, ctx->n_total_clusters * sizeof(float));
    
    // 计算 Norm（此时数据已全部在 GPU）
    compute_l2_norm_gpu(ctx->d_cluster_vectors, ctx->d_cluster_vector_norm, ctx->n_total_vectors, ctx->n_dim, L2NORM_AUTO, 0);
    compute_l2_norm_gpu(ctx->d_cluster_centers, ctx->d_cluster_centers_norm, ctx->n_total_clusters, ctx->n_dim, L2NORM_AUTO, 0);
    
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    
    ctx->is_initialized = true;
}

// ---------------------------------------------------------
// 2. Query Batch 上下文管理
// ---------------------------------------------------------

void* ivf_create_batch_context(int max_n_query, int n_dim, int max_n_probes, int max_k, int n_total_clusters) {
    IVFQueryBatchContext* ctx = new IVFQueryBatchContext();
    
    ctx->max_n_query = max_n_query;
    ctx->n_dim = n_dim;
    ctx->max_n_probes = max_n_probes;
    ctx->max_k = max_k;
    ctx->n_total_clusters = n_total_clusters;
    
    cudaStream_t stream;
    cudaEvent_t data_ready_event;
    cudaEvent_t compute_done_event;
    
    cudaStreamCreate(&stream);
    cudaEventCreate(&data_ready_event);
    cudaEventCreate(&compute_done_event);
    
    ctx->stream = stream;
    ctx->data_ready_event = data_ready_event;
    ctx->compute_done_event = compute_done_event;
    
    // 预分配所有需要的显存，避免在循环中malloc
    cudaMalloc(&ctx->d_queries, max_n_query * n_dim * sizeof(float));
    cudaMalloc(&ctx->d_query_norm, max_n_query * sizeof(float));
    
    // 中间结果
    cudaMalloc(&ctx->d_topk_dist, max_n_query * max_k * sizeof(float));
    cudaMalloc(&ctx->d_topk_index, max_n_query * max_k * sizeof(int));
    cudaMalloc(&ctx->d_top_nprobe_index, max_n_query * max_n_probes * sizeof(int));
    cudaMalloc(&ctx->d_top_nprobe_dist, max_n_query * max_n_probes * sizeof(float));
    cudaMalloc(&ctx->d_inner_product, max_n_query * n_total_clusters * sizeof(float));
    
    // 粗筛索引
    cudaMalloc(&ctx->d_index_seq, max_n_query * n_total_clusters * sizeof(int));
    
    // Entry构建相关
    cudaMalloc(&ctx->d_cluster_query_count, n_total_clusters * sizeof(int));
    cudaMalloc(&ctx->d_cluster_query_offset, (n_total_clusters + 1) * sizeof(int));
    cudaMalloc(&ctx->d_entry_count_per_cluster, n_total_clusters * sizeof(int));
    cudaMalloc(&ctx->d_entry_offset, (n_total_clusters + 1) * sizeof(int));
    cudaMalloc(&ctx->d_cluster_write_pos, n_total_clusters * sizeof(int));
    cudaMalloc(&ctx->d_entry_query_offset, (n_total_clusters + 1) * sizeof(int));
    
    // 这些根据 query * probes 动态变化的，先按最大值估算分配
    size_t max_entries = max_n_query * max_n_probes;
    cudaMalloc(&ctx->d_cluster_query_data, max_entries * sizeof(int));
    cudaMalloc(&ctx->d_cluster_query_probe_indices, max_entries * sizeof(int));
    
    // Entry Arrays
    cudaMalloc(&ctx->d_entry_cluster_id, max_entries * sizeof(int));
    cudaMalloc(&ctx->d_entry_query_start, max_entries * sizeof(int));
    cudaMalloc(&ctx->d_entry_query_count, max_entries * sizeof(int));
    cudaMalloc(&ctx->d_entry_queries, max_entries * sizeof(int));
    cudaMalloc(&ctx->d_entry_probe_indices, max_entries * sizeof(int));
    
    // Fine Search Candidates
    cudaMalloc(&ctx->d_topk_dist_candidate, max_n_query * max_n_probes * max_k * sizeof(float));
    cudaMalloc(&ctx->d_topk_index_candidate, max_n_query * max_n_probes * max_k * sizeof(int));
    
    CHECK_CUDA_ERRORS;
    
    return ctx;
}

void ivf_destroy_batch_context(void* ctx_ptr) {
    if(!ctx_ptr) return;
    IVFQueryBatchContext* ctx = (IVFQueryBatchContext*)ctx_ptr;
    
    cudaStreamDestroy(get_stream(ctx));
    cudaEventDestroy(get_data_ready_event(ctx));
    cudaEventDestroy(get_compute_done_event(ctx));
    
    cudaFree(ctx->d_queries);
    cudaFree(ctx->d_query_norm);
    cudaFree(ctx->d_topk_dist);
    cudaFree(ctx->d_topk_index);
    cudaFree(ctx->d_top_nprobe_index);
    cudaFree(ctx->d_top_nprobe_dist);
    cudaFree(ctx->d_inner_product);
    cudaFree(ctx->d_index_seq);
    
    cudaFree(ctx->d_cluster_query_count);
    cudaFree(ctx->d_cluster_query_offset);
    cudaFree(ctx->d_entry_count_per_cluster);
    cudaFree(ctx->d_entry_offset);
    cudaFree(ctx->d_cluster_write_pos);
    cudaFree(ctx->d_entry_query_offset);
    
    cudaFree(ctx->d_cluster_query_data);
    cudaFree(ctx->d_cluster_query_probe_indices);
    cudaFree(ctx->d_entry_cluster_id);
    cudaFree(ctx->d_entry_query_start);
    cudaFree(ctx->d_entry_query_count);
    cudaFree(ctx->d_entry_queries);
    cudaFree(ctx->d_entry_probe_indices);
    cudaFree(ctx->d_topk_dist_candidate);
    cudaFree(ctx->d_topk_index_candidate);
    
    delete ctx;
}

// ---------------------------------------------------------
// 3. Pipeline 阶段分解
// ---------------------------------------------------------

/**
 * 阶段 1: 数据预处理 (Preprocessing)
 * 职责：将Query数据上传GPU，并计算Norm。
 * 特点：利用DMA传输，不占用太多GPU Compute。
 */
void ivf_pipeline_stage1_prepare(
    void* batch_ctx_ptr,
    float* query_batch_host, // 连续存储 [n_query * n_dim]
    int n_query
) {
    IVFQueryBatchContext* ctx = (IVFQueryBatchContext*)batch_ctx_ptr;
    
    if (n_query > ctx->max_n_query) {
        fprintf(stderr, "[Error] Batch size exceeds capacity: %d > %d\n", n_query, ctx->max_n_query);
        return;
    }
    
    cudaStream_t stream = get_stream(ctx);
    
    // 1. 异步上传 Query
    cudaMemcpyAsync(ctx->d_queries, query_batch_host, 
                    n_query * ctx->n_dim * sizeof(float), 
                    cudaMemcpyHostToDevice, stream);
    
    // 2. 计算 Query Norm
    compute_l2_norm_gpu(ctx->d_queries, ctx->d_query_norm, n_query, ctx->n_dim, L2NORM_AUTO, stream);
    
    // 3. 记录事件：数据准备完毕
    cudaEventRecord(get_data_ready_event(ctx), stream);
}

/**
 * 阶段 2: 核心计算 (Compute)
 * 职责：执行粗筛和精筛。
 * 特点：计算密集。
 */
void ivf_pipeline_stage2_compute(
    void* batch_ctx_ptr,
    void* idx_ctx_ptr,
    int n_query,
    int n_probes,
    int k,
    int distance_mode
) {
    /* 安全检查：空指针检查 */
    if (!batch_ctx_ptr) {
        throw std::invalid_argument("ivf_pipeline_stage2_compute: batch_ctx_ptr is NULL");
    }
    if (!idx_ctx_ptr) {
        throw std::invalid_argument("ivf_pipeline_stage2_compute: idx_ctx_ptr is NULL");
    }
    
    IVFQueryBatchContext* q_ctx = (IVFQueryBatchContext*)batch_ctx_ptr;
    IVFIndexContext* idx_ctx = (IVFIndexContext*)idx_ctx_ptr;
    
    /* 安全检查：索引上下文初始化状态 */
    if (!idx_ctx->is_initialized) {
        throw std::runtime_error("ivf_pipeline_stage2_compute: 索引上下文未初始化 (is_initialized=false)");
    }
    
    /* 安全检查：关键 GPU 内存指针 */
    if (!idx_ctx->d_cluster_centers) {
        throw std::runtime_error("ivf_pipeline_stage2_compute: d_cluster_centers is NULL");
    }
    if (!idx_ctx->d_cluster_centers_norm) {
        throw std::runtime_error("ivf_pipeline_stage2_compute: d_cluster_centers_norm is NULL");
    }
    if (!idx_ctx->d_cluster_vectors) {
        throw std::runtime_error("ivf_pipeline_stage2_compute: d_cluster_vectors is NULL");
    }
    if (!idx_ctx->d_cluster_vector_norm) {
        throw std::runtime_error("ivf_pipeline_stage2_compute: d_cluster_vector_norm is NULL");
    }
    if (!idx_ctx->d_probe_vector_offset) {
        throw std::runtime_error("ivf_pipeline_stage2_compute: d_probe_vector_offset is NULL");
    }
    if (!idx_ctx->d_probe_vector_count) {
        throw std::runtime_error("ivf_pipeline_stage2_compute: d_probe_vector_count is NULL");
    }
    
    /* 安全检查：参数有效性 */
    if (idx_ctx->n_total_clusters <= 0) {
        throw std::invalid_argument("ivf_pipeline_stage2_compute: n_total_clusters <= 0");
    }
    if (idx_ctx->n_dim <= 0) {
        throw std::invalid_argument("ivf_pipeline_stage2_compute: n_dim <= 0");
    }
    if (n_probes > idx_ctx->n_total_clusters) {
        throw std::invalid_argument("ivf_pipeline_stage2_compute: n_probes > n_total_clusters");
    }
    
    /* 检查 CUDA 错误状态 */
    CHECK_CUDA_ERRORS;
    
    cudaStream_t stream = get_stream(q_ctx);
    
    // ---------------- Coarse Search ----------------
    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    
    // 生成序列索引 (0,1,2...) 用于粗筛
    dim3 queryDim(n_query);
    dim3 block_dim(std::min(idx_ctx->n_total_clusters, 256));
    generate_sequence_indices_kernel<<<queryDim, block_dim, 0, stream>>>(
        q_ctx->d_index_seq, n_query, idx_ctx->n_total_clusters);
    
    // 初始化距离
    dim3 fill_block(256);
    int fill_grid_size = (n_query * n_probes + fill_block.x - 1) / fill_block.x;
    dim3 fill_grid(fill_grid_size);
    fill_kernel<<<fill_grid, fill_block, 0, stream>>>(
        q_ctx->d_top_nprobe_dist, FLT_MAX, n_query * n_probes);
    
    // 计算 Inner Product (Query x Centers)
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                idx_ctx->n_total_clusters, n_query, idx_ctx->n_dim,
                &alpha,
                idx_ctx->d_cluster_centers, idx_ctx->n_dim,
                q_ctx->d_queries, idx_ctx->n_dim,
                &beta,
                q_ctx->d_inner_product, idx_ctx->n_total_clusters);
    
    // TopK Cosine
    pgvector::fusion_dist_topk_warpsort::fusion_cos_topk_warpsort<float, int>(
        q_ctx->d_query_norm, idx_ctx->d_cluster_centers_norm, q_ctx->d_inner_product, q_ctx->d_index_seq,
        n_query, idx_ctx->n_total_clusters, n_probes,
        q_ctx->d_top_nprobe_dist, q_ctx->d_top_nprobe_index,
        true, stream
    );
    
    cublasDestroy(handle);
    
    // ---------------- Build Entry Data ----------------
    // 清零 Cluster Count
    cudaMemsetAsync(q_ctx->d_cluster_query_count, 0, idx_ctx->n_total_clusters * sizeof(int), stream);
    
    // 统计 Query 分布
    dim3 probeDim(n_probes);
    count_cluster_queries_kernel<<<queryDim, probeDim, 0, stream>>>(
        q_ctx->d_top_nprobe_index, q_ctx->d_cluster_query_count, n_query, n_probes, idx_ctx->n_total_clusters
    );
    
    // 前缀和计算 offset
    compute_prefix_sum(q_ctx->d_cluster_query_count, q_ctx->d_cluster_query_offset, idx_ctx->n_total_clusters, stream);
    
    // 构建 Cluster -> Query 映射
    cudaMemcpyAsync(q_ctx->d_cluster_write_pos, q_ctx->d_cluster_query_offset, 
                    idx_ctx->n_total_clusters * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    
    build_cluster_query_mapping_kernel<<<queryDim, probeDim, 0, stream>>>(
        q_ctx->d_top_nprobe_index, q_ctx->d_cluster_query_offset, 
        q_ctx->d_cluster_query_data, q_ctx->d_cluster_query_probe_indices, 
        q_ctx->d_cluster_write_pos, n_query, n_probes, idx_ctx->n_total_clusters
    );
    
    // 计算 Entry 数量
    constexpr int kQueriesPerBlock = 8;
    dim3 clusterDim(idx_ctx->n_total_clusters);
    dim3 blockDim_entry(1);
    count_entries_per_cluster_kernel<<<clusterDim, blockDim_entry, 0, stream>>>(
        q_ctx->d_cluster_query_offset, q_ctx->d_entry_count_per_cluster, idx_ctx->n_total_clusters, kQueriesPerBlock
    );
    
    compute_prefix_sum(q_ctx->d_entry_count_per_cluster, q_ctx->d_entry_offset, idx_ctx->n_total_clusters, stream);
    
    // 获取总 Entry 数
    int n_entry = 0;
    cudaMemcpyAsync(&n_entry, q_ctx->d_entry_offset + idx_ctx->n_total_clusters, sizeof(int), cudaMemcpyDeviceToHost, stream);
    
    // 需要同步一下获取 n_entry，这会轻微阻塞 Host，但对流水线影响可控
    cudaStreamSynchronize(stream);
    
    if (n_entry > 0) {
        // 构建 Entry
        cudaMemcpyAsync(q_ctx->d_entry_query_offset, q_ctx->d_cluster_query_offset, 
                        (idx_ctx->n_total_clusters + 1) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        
        build_entry_data_kernel<<<clusterDim, blockDim_entry, 0, stream>>>(
            q_ctx->d_cluster_query_offset, q_ctx->d_cluster_query_data, q_ctx->d_cluster_query_probe_indices,
            q_ctx->d_entry_offset, q_ctx->d_entry_query_offset,
            q_ctx->d_entry_cluster_id, q_ctx->d_entry_query_start, q_ctx->d_entry_query_count,
            q_ctx->d_entry_queries, q_ctx->d_entry_probe_indices,
            idx_ctx->n_total_clusters, kQueriesPerBlock
        );
        
        // ---------------- Fine Search ----------------
        // 初始化无效值
        dim3 init_block(512);
        int init_grid_size = (n_query * n_probes * k + init_block.x - 1) / init_block.x;
        dim3 init_grid(init_grid_size);
        init_invalid_values_kernel<<<init_grid, init_block, 0, stream>>>(
            q_ctx->d_topk_dist_candidate, q_ctx->d_topk_index_candidate, n_query * n_probes * k
        );
        
        // 选择合适的 Kernel 变体
        int capacity = 32;
        while (capacity < k) capacity <<= 1;
        capacity = std::min(capacity, kMaxCapacity);
        
        dim3 block(kQueriesPerBlock * 32);
        
        // 根据capacity选择kernel实例
        if (distance_mode == COSINE_DISTANCE){
            if (capacity <= 32) {
                launch_indexed_inner_product_with_cos_topk_kernel<64, true, kQueriesPerBlock>(
                    block, idx_ctx->n_dim, q_ctx->d_queries,
                    idx_ctx->d_cluster_vectors, idx_ctx->d_probe_vector_offset, idx_ctx->d_probe_vector_count,
                    q_ctx->d_entry_cluster_id, q_ctx->d_entry_query_start, q_ctx->d_entry_query_count,
                    q_ctx->d_entry_queries, q_ctx->d_entry_probe_indices,
                    q_ctx->d_query_norm, idx_ctx->d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    q_ctx->d_topk_dist_candidate, q_ctx->d_topk_index_candidate, stream
                );
            } else if (capacity <= 64) {
                launch_indexed_inner_product_with_cos_topk_kernel<128, true, kQueriesPerBlock>(
                    block, idx_ctx->n_dim, q_ctx->d_queries,
                    idx_ctx->d_cluster_vectors, idx_ctx->d_probe_vector_offset, idx_ctx->d_probe_vector_count,
                    q_ctx->d_entry_cluster_id, q_ctx->d_entry_query_start, q_ctx->d_entry_query_count,
                    q_ctx->d_entry_queries, q_ctx->d_entry_probe_indices,
                    q_ctx->d_query_norm, idx_ctx->d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    q_ctx->d_topk_dist_candidate, q_ctx->d_topk_index_candidate, stream
                );
            } else {
                launch_indexed_inner_product_with_cos_topk_kernel<256, true, kQueriesPerBlock>(
                    block, idx_ctx->n_dim, q_ctx->d_queries,
                    idx_ctx->d_cluster_vectors, idx_ctx->d_probe_vector_offset, idx_ctx->d_probe_vector_count,
                    q_ctx->d_entry_cluster_id, q_ctx->d_entry_query_start, q_ctx->d_entry_query_count,
                    q_ctx->d_entry_queries, q_ctx->d_entry_probe_indices,
                    q_ctx->d_query_norm, idx_ctx->d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    q_ctx->d_topk_dist_candidate, q_ctx->d_topk_index_candidate, stream
                );
            }
        }
        else if(distance_mode == L2_DISTANCE){
            if (capacity <= 32) {
                launch_indexed_inner_product_with_l2_topk_kernel<64, true, kQueriesPerBlock>(
                    block, idx_ctx->n_dim, q_ctx->d_queries,
                    idx_ctx->d_cluster_vectors, idx_ctx->d_probe_vector_offset, idx_ctx->d_probe_vector_count,
                    q_ctx->d_entry_cluster_id, q_ctx->d_entry_query_start, q_ctx->d_entry_query_count,
                    q_ctx->d_entry_queries, q_ctx->d_entry_probe_indices,
                    q_ctx->d_query_norm, idx_ctx->d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    q_ctx->d_topk_dist_candidate, q_ctx->d_topk_index_candidate, stream
                );
            } else if (capacity <= 64) {
                launch_indexed_inner_product_with_l2_topk_kernel<128, true, kQueriesPerBlock>(
                    block, idx_ctx->n_dim, q_ctx->d_queries,
                    idx_ctx->d_cluster_vectors, idx_ctx->d_probe_vector_offset, idx_ctx->d_probe_vector_count,
                    q_ctx->d_entry_cluster_id, q_ctx->d_entry_query_start, q_ctx->d_entry_query_count,
                    q_ctx->d_entry_queries, q_ctx->d_entry_probe_indices,
                    q_ctx->d_query_norm, idx_ctx->d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    q_ctx->d_topk_dist_candidate, q_ctx->d_topk_index_candidate, stream
                );
            } else {
                launch_indexed_inner_product_with_l2_topk_kernel<256, true, kQueriesPerBlock>(
                    block, idx_ctx->n_dim, q_ctx->d_queries,
                    idx_ctx->d_cluster_vectors, idx_ctx->d_probe_vector_offset, idx_ctx->d_probe_vector_count,
                    q_ctx->d_entry_cluster_id, q_ctx->d_entry_query_start, q_ctx->d_entry_query_count,
                    q_ctx->d_entry_queries, q_ctx->d_entry_probe_indices,
                    q_ctx->d_query_norm, idx_ctx->d_cluster_vector_norm,
                    n_entry, n_probes, k,
                    q_ctx->d_topk_dist_candidate, q_ctx->d_topk_index_candidate, stream
                );
            }
        }

        
        // ---------------- Selection & Mapping ----------------
        select_k<float, int>(q_ctx->d_topk_dist_candidate, n_query, n_probes * k, k,
                             q_ctx->d_topk_dist, q_ctx->d_topk_index, true, stream);
        
        dim3 map_block(256);
        dim3 map_grid((n_query * k + map_block.x - 1) / map_block.x);
        map_candidate_indices_kernel<<<map_grid, map_block, 0, stream>>>(
            q_ctx->d_topk_index_candidate, q_ctx->d_topk_index, n_query, n_probes, k
        );
    }
    
    // 标记计算完成
    cudaEventRecord(get_compute_done_event(q_ctx), stream);
}

/**
 * 获取结果 (Download)
 * 职责：将结果传回CPU
 */
void ivf_pipeline_get_results(
    void* batch_ctx_ptr,
    float* topk_dist,
    int* topk_index,
    int n_query,
    int k
) {
    IVFQueryBatchContext* ctx = (IVFQueryBatchContext*)batch_ctx_ptr;
    cudaStream_t stream = get_stream(ctx);
    
    cudaMemcpyAsync(topk_dist, ctx->d_topk_dist, n_query * k * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(topk_index, ctx->d_topk_index, n_query * k * sizeof(int), cudaMemcpyDeviceToHost, stream);
    
    // 必须同步，因为pgvector拿到数据后马上要用
    cudaStreamSynchronize(stream);
}

/**
 * 同步 (Wait)
 * 用于 Host 等待当前流完成
 */
void ivf_pipeline_sync_batch(void* batch_ctx_ptr) {
    IVFQueryBatchContext* ctx = (IVFQueryBatchContext*)batch_ctx_ptr;
    cudaStream_t stream = get_stream(ctx);
    cudaStreamSynchronize(stream);
}

