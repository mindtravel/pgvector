#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

#include <stdbool.h>
#include <stddef.h>  // for size_t

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * 缓冲区容量配置 
 * 例如：64K 向量 * 1024 维 * 4 bytes = 256MB per chunk
 * 可根据显存和 CPU 内存情况调整 
 */
#define PIPELINE_CHUNK_SIZE 65536 

/* 
 * 严格兼容 C 和 C++ 的上下文结构 
 * 所有 CUDA 特有类型（如 cudaEvent_t）都用 void* 隐藏
 */
typedef struct GpuPipelineContext {
    /* --- 配置 --- */
    int         dimensions;
    size_t      chunk_capacity;
    
    /* --- 状态 --- */
    size_t      total_uploaded;
    
    /* --- 双缓冲资源 (Double Buffering) --- */
    /* CPU Pinned Memory Buffers (Host) */
    float*      h_vec_buffers[2];   
    
    /* Buffer 当前填充量 */
    size_t      current_counts[2];
    
    /* 当前活跃的 Buffer 索引 (0 或 1) - CPU 正在写的那个 */
    int         active_buf_idx;
    
    /* CUDA Events (隐藏为 void*) - 用于同步 */
    void*       events[2]; 
    
    /* --- GPU 目标地址 --- */
    float*      d_vectors_base;
    
} GpuPipelineContext;

/*
* 测试cuda是否可用
*/ 
extern bool cuda_is_available(void);

/* GPU 内存管理函数 */
extern void** cuda_malloc(void** d_ptr, size_t size);
extern void* cuda_alloc_pinned(size_t size);
extern void cuda_free_pinned(void* ptr);
extern void cuda_free(void* d_ptr);
extern void cuda_memcpy_h2d(void* d_dst, const void* h_src, size_t size);
extern void cuda_memcpy_d2h(void* h_dst, const void* d_src, size_t size);
extern void cuda_memcpy_async_h2d(void* d_dst, const void* h_src, size_t size);
extern void cuda_memcpy_async_d2h(void* h_dst, const void* d_src, size_t size);

/* 辅助清理函数 */
extern void cuda_cleanup_memory(float* d_query_batch, int* d_cluster_size, float* d_cluster_vectors,
                                float* d_cluster_centers, int* d_initial_indices, float* d_topk_dist, int* d_topk_index);

/* 流水线辅助函数*/
extern void cuda_pipeline_init(GpuPipelineContext* ctx, int dim, size_t total_vectors, 
    float* d_vec_ptr);
extern void cuda_pipeline_flush(GpuPipelineContext* ctx);
extern void cuda_pipeline_flush_vectors_only(GpuPipelineContext* ctx);
extern void cuda_pipeline_free(GpuPipelineContext* ctx);

/* CUDA 错误检查函数 */
extern void cuda_device_synchronize(void);
extern const char* cuda_get_last_error_string(void);
extern bool cuda_check_last_error(void);


#ifdef __cplusplus
}
#endif

#endif /* CUDA_WRAPPER_H */