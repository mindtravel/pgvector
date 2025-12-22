#include <stdexcept>
#include <cstdio>
#include <cstring>
#include "integrate_screen.cuh"

extern "C" {
/**
 * C包装函数，用于从C代码调用batch_search_pipeline（新接口：接受 device 指针）
 * 处理C++异常并返回错误码
 */
int batch_search_pipeline_wrapper(float* d_query_batch,
                                  int* d_cluster_size,
                                  float* d_cluster_vectors,
                                  float* d_cluster_centers,
                                  int* d_initial_indices,
                                  float* d_topk_dist,
                                  int* d_topk_index,
                                  int n_query, int n_dim, int n_total_cluster,
                                  int n_total_vectors, int n_probes, int k, int distance_mode)
{
    /* 记录关键参数信息到PostgreSQL日志 */
    fprintf(stderr, "batch_search_pipeline_wrapper: ========== 参数验证 ==========\n");
    fprintf(stderr, "batch_search_pipeline_wrapper: n_query=%d, n_dim=%d, n_total_cluster=%d, n_total_vectors=%d, n_probes=%d, k=%d\n",
            n_query, n_dim, n_total_cluster, n_total_vectors, n_probes, k);
    
    /* 验证GPU内存指针 */
    fprintf(stderr, "batch_search_pipeline_wrapper: GPU内存指针:\n");
    fprintf(stderr, "batch_search_pipeline_wrapper:   d_query_batch=%p (expected size: %zu bytes)\n",
            (void*)d_query_batch, (size_t)n_query * n_dim * sizeof(float));
    fprintf(stderr, "batch_search_pipeline_wrapper:   d_cluster_size=%p (expected size: %zu bytes)\n",
            (void*)d_cluster_size, (size_t)n_total_cluster * sizeof(int));
    fprintf(stderr, "batch_search_pipeline_wrapper:   d_cluster_vectors=%p (expected size: %zu bytes)\n",
            (void*)d_cluster_vectors, (size_t)n_total_vectors * n_dim * sizeof(float));
    fprintf(stderr, "batch_search_pipeline_wrapper:   d_cluster_centers=%p (expected size: %zu bytes)\n",
            (void*)d_cluster_centers, (size_t)n_total_cluster * n_dim * sizeof(float));
    fprintf(stderr, "batch_search_pipeline_wrapper:   d_initial_indices=%p (expected size: %zu bytes)\n",
            (void*)d_initial_indices, d_initial_indices ? (size_t)n_query * n_total_cluster * sizeof(int) : 0);
    fprintf(stderr, "batch_search_pipeline_wrapper:   d_topk_dist=%p (expected size: %zu bytes)\n",
            (void*)d_topk_dist, (size_t)n_query * k * sizeof(float));
    fprintf(stderr, "batch_search_pipeline_wrapper:   d_topk_index=%p (expected size: %zu bytes)\n",
            (void*)d_topk_index, (size_t)n_query * k * sizeof(int));
    
    /* 验证参数有效性 */
    if (!d_query_batch || !d_cluster_size || !d_cluster_vectors || 
        !d_cluster_centers || !d_topk_dist || !d_topk_index) {
        fprintf(stderr, "batch_search_pipeline_wrapper: 错误: GPU内存指针为NULL\n");
        return -1;
    }
    
    if (n_query <= 0 || n_dim <= 0 || n_total_cluster <= 0 || 
        n_total_vectors <= 0 || n_probes <= 0 || k <= 0) {
        fprintf(stderr, "batch_search_pipeline_wrapper: 错误: 参数值无效\n");
        return -1;
    }
    
    if (n_probes > n_total_cluster) {
        fprintf(stderr, "batch_search_pipeline_wrapper: 警告: n_probes(%d) > n_total_cluster(%d)\n",
                n_probes, n_total_cluster);
    }
    
    fprintf(stderr, "batch_search_pipeline_wrapper: ========== 调用 batch_search_pipeline ==========\n");
    
    try {
        batch_search_pipeline(d_query_batch, d_cluster_size, d_cluster_vectors, d_cluster_centers,
                             d_initial_indices,  // 传入初始索引（如果为nullptr，则内部生成）
                             d_topk_dist, d_topk_index,
                             n_query, n_dim, n_total_cluster, n_total_vectors, n_probes, k, distance_mode);
        fprintf(stderr, "batch_search_pipeline_wrapper: 执行成功\n");
        return 0;
    } catch (const std::exception& e) {
        /* 记录异常信息到PostgreSQL日志 */
        fprintf(stderr, "batch_search_pipeline_wrapper: 捕获到异常 - %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "batch_search_pipeline_wrapper: 捕获到未知异常\n");
        return -1;
    }
}

/* 
 * 分离流水线函数的 C wrapper
 * 这些函数从 C 代码调用，处理 C++ 异常
 */

void* ivf_create_index_context_wrapper(void) {
    try {
        return ivf_create_index_context();
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_create_index_context_wrapper: 异常 - %s\n", e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "ivf_create_index_context_wrapper: 未知异常\n");
        return nullptr;
    }
}

void ivf_destroy_index_context_wrapper(void* ctx_ptr) {
    try {
        ivf_destroy_index_context(ctx_ptr);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_destroy_index_context_wrapper: 异常 - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_destroy_index_context_wrapper: 未知异常\n");
    }
}

int ivf_load_dataset_wrapper(
    void* idx_ctx_ptr,
    int* d_cluster_size,
    float* d_cluster_vectors,
    float* d_cluster_centers,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
) {
    try {
        return ivf_load_dataset(idx_ctx_ptr, d_cluster_size, d_cluster_vectors, d_cluster_centers,
                               n_total_clusters, n_total_vectors, n_dim);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_load_dataset_wrapper: 异常 - %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "ivf_load_dataset_wrapper: 未知异常\n");
        return -1;
    }
}

void* ivf_create_batch_context_wrapper(int max_n_query, int n_dim, int max_n_probes, int max_k, int n_total_clusters) {
    try {
        return ivf_create_batch_context(max_n_query, n_dim, max_n_probes, max_k, n_total_clusters);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_create_batch_context_wrapper: 异常 - %s\n", e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "ivf_create_batch_context_wrapper: 未知异常\n");
        return nullptr;
    }
}

void ivf_destroy_batch_context_wrapper(void* ctx_ptr) {
    try {
        ivf_destroy_batch_context(ctx_ptr);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_destroy_batch_context_wrapper: 异常 - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_destroy_batch_context_wrapper: 未知异常\n");
    }
}

void ivf_pipeline_stage1_prepare_wrapper(
    void* batch_ctx_ptr,
    float* query_batch_host,
    int n_query
) {
    try {
        ivf_pipeline_stage1_prepare(batch_ctx_ptr, query_batch_host, n_query);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_pipeline_stage1_prepare_wrapper: 异常 - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_pipeline_stage1_prepare_wrapper: 未知异常\n");
    }
}

/* 检查索引上下文是否已初始化 */
int ivf_check_index_initialized_wrapper(void* idx_ctx_ptr) {
    if (!idx_ctx_ptr) {
        fprintf(stderr, "ivf_check_index_initialized_wrapper: idx_ctx_ptr is NULL\n");
        return 0;
    }
    try {
        /* 注意：这里需要访问 IVFIndexContext 结构，但由于是 C++ 代码，可以直接访问 */
        /* 为了避免包含头文件，我们通过函数调用检查 */
        /* 实际检查在 ivf_pipeline_stage2_compute 中进行 */
        return 1; /* 返回1表示指针非空，实际初始化状态在 compute 函数中检查 */
    } catch (...) {
        fprintf(stderr, "ivf_check_index_initialized_wrapper: 检查失败\n");
        return 0;
    }
}

void ivf_pipeline_stage2_compute_wrapper(
    void* batch_ctx_ptr,
    void* idx_ctx_ptr,
    int n_query,
    int n_probes,
    int k,
    int distance_mode
) {
    /* 参数验证 */
    if (!batch_ctx_ptr) {
        fprintf(stderr, "ivf_pipeline_stage2_compute_wrapper: batch_ctx_ptr is NULL\n");
        return;
    }
    if (!idx_ctx_ptr) {
        fprintf(stderr, "ivf_pipeline_stage2_compute_wrapper: idx_ctx_ptr is NULL\n");
        return;
    }
    if (n_query <= 0 || n_probes <= 0 || k <= 0) {
        fprintf(stderr, "ivf_pipeline_stage2_compute_wrapper: 无效参数 - n_query=%d, n_probes=%d, k=%d\n",
                n_query, n_probes, k);
        return;
    }
    
    try {
        ivf_pipeline_stage2_compute(batch_ctx_ptr, idx_ctx_ptr, n_query, n_probes, k, distance_mode);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_pipeline_stage2_compute_wrapper: 异常 - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_pipeline_stage2_compute_wrapper: 未知异常\n");
    }
}

void ivf_pipeline_get_results_wrapper(
    void* batch_ctx_ptr,
    float* topk_dist,
    int* topk_index,
    int n_query,
    int k
) {
    try {
        ivf_pipeline_get_results(batch_ctx_ptr, topk_dist, topk_index, n_query, k);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_pipeline_get_results_wrapper: 异常 - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_pipeline_get_results_wrapper: 未知异常\n");
    }
}

void ivf_pipeline_sync_batch_wrapper(void* batch_ctx_ptr) {
    try {
        ivf_pipeline_sync_batch(batch_ctx_ptr);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_pipeline_sync_batch_wrapper: 异常 - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_pipeline_sync_batch_wrapper: 未知异常\n");
    }
}

/* 流式上传函数的 C wrapper */
void ivf_init_streaming_upload_wrapper(
    void* idx_ctx_ptr,
    int n_total_clusters,
    int n_total_vectors,
    int n_dim
) {
    try {
        ivf_init_streaming_upload(idx_ctx_ptr, n_total_clusters, n_total_vectors, n_dim);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_init_streaming_upload_wrapper: 异常 - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_init_streaming_upload_wrapper: 未知异常\n");
    }
}

void ivf_append_cluster_data_wrapper(
    void* idx_ctx_ptr,
    int cluster_id,
    float* host_vector_data,
    int count,
    int start_offset_idx
) {
    try {
        ivf_append_cluster_data(idx_ctx_ptr, cluster_id, host_vector_data, count, start_offset_idx);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_append_cluster_data_wrapper: 异常 - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_append_cluster_data_wrapper: 未知异常\n");
    }
}

void ivf_finalize_streaming_upload_wrapper(
    void* idx_ctx_ptr,
    float* center_data_flat,
    int total_vectors_check
) {
    try {
        ivf_finalize_streaming_upload(idx_ctx_ptr, center_data_flat, total_vectors_check);
    } catch (const std::exception& e) {
        fprintf(stderr, "ivf_finalize_streaming_upload_wrapper: 异常 - %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "ivf_finalize_streaming_upload_wrapper: 未知异常\n");
    }
}

/* ========================================================================= */
/*                              索引句柄注册表                                */
/* ========================================================================= */

/* 简单的注册表实现（进程内，Session级别） */
#define MAX_GPU_INDICES 64

typedef struct {
    unsigned int index_oid;  /* Oid 类型在 C++ 中通常映射为 unsigned int */
    void* handle;
    bool active;
} GpuIndexEntry;

static GpuIndexEntry g_gpu_indices[MAX_GPU_INDICES] = {0};
static int g_gpu_indices_count = 0;

/**
 * 注册索引实例到全局注册表
 * 
 * @param index_oid 索引的 OID
 * @param gpu_handle GPU 句柄
 */
void ivf_register_index_instance(unsigned int index_oid, void* gpu_handle) {
    if (gpu_handle == NULL) {
        fprintf(stderr, "ivf_register_index_instance: gpu_handle 为 NULL\n");
        return;
    }
    
    /* 查找是否已存在该 OID 的条目 */
    for (int i = 0; i < g_gpu_indices_count; i++) {
        if (g_gpu_indices[i].active && g_gpu_indices[i].index_oid == index_oid) {
            /* 如果已存在，先释放旧的句柄 */
            if (g_gpu_indices[i].handle != NULL) {
                ivf_destroy_index_context_wrapper(g_gpu_indices[i].handle);
            }
            g_gpu_indices[i].handle = gpu_handle;
            fprintf(stderr, "ivf_register_index_instance: 更新索引 OID %u 的句柄\n", index_oid);
            return;
        }
    }
    
    /* 查找空位或添加新条目 */
    if (g_gpu_indices_count < MAX_GPU_INDICES) {
        g_gpu_indices[g_gpu_indices_count].index_oid = index_oid;
        g_gpu_indices[g_gpu_indices_count].handle = gpu_handle;
        g_gpu_indices[g_gpu_indices_count].active = true;
        g_gpu_indices_count++;
        fprintf(stderr, "ivf_register_index_instance: 注册索引 OID %u，当前注册数: %d\n", index_oid, g_gpu_indices_count);
    } else {
        /* 查找是否有非活跃的条目可以重用 */
        for (int i = 0; i < MAX_GPU_INDICES; i++) {
            if (!g_gpu_indices[i].active) {
                g_gpu_indices[i].index_oid = index_oid;
                g_gpu_indices[i].handle = gpu_handle;
                g_gpu_indices[i].active = true;
                fprintf(stderr, "ivf_register_index_instance: 重用槽位 %d 注册索引 OID %u\n", i, index_oid);
                return;
            }
        }
        fprintf(stderr, "ivf_register_index_instance: 错误 - 注册表已满（最大 %d 个索引）\n", MAX_GPU_INDICES);
    }
}

/**
 * 根据索引 OID 获取 GPU 句柄
 * 
 * @param index_oid 索引的 OID
 * @return GPU 句柄，如果不存在则返回 NULL
 */
void* ivf_get_index_instance(unsigned int index_oid) {
    for (int i = 0; i < g_gpu_indices_count; i++) {
        if (g_gpu_indices[i].active && g_gpu_indices[i].index_oid == index_oid) {
            return g_gpu_indices[i].handle;
        }
    }
    
    /* 也检查所有槽位（包括非活跃的，以防 count 不准确） */
    for (int i = 0; i < MAX_GPU_INDICES; i++) {
        if (g_gpu_indices[i].active && g_gpu_indices[i].index_oid == index_oid) {
            return g_gpu_indices[i].handle;
        }
    }
    
    return NULL;
}

/**
 * 注销索引实例（可选，用于清理）
 * 
 * @param index_oid 索引的 OID
 */
void ivf_unregister_index_instance(unsigned int index_oid) {
    for (int i = 0; i < MAX_GPU_INDICES; i++) {
        if (g_gpu_indices[i].active && g_gpu_indices[i].index_oid == index_oid) {
            if (g_gpu_indices[i].handle != NULL) {
                ivf_destroy_index_context_wrapper(g_gpu_indices[i].handle);
            }
            g_gpu_indices[i].active = false;
            g_gpu_indices[i].handle = NULL;
            fprintf(stderr, "ivf_unregister_index_instance: 注销索引 OID %u\n", index_oid);
            return;
        }
    }
}
}

