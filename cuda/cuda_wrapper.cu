/**
 * cuda_wrapper.cu
 */ 
#include "pch.h"
#include "cuda_wrapper.h"

#include <cuda_runtime.h>

/**
 * 包装C语言调用cuda函数的接口
 **/
extern "C" {
    /*
    * 检查CUDA是否可用
    */  
    bool cuda_is_available() {
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            printf("DEBUG: cudaGetDeviceCount 失败: %s\n", cudaGetErrorString(err));
            return false;
        }
        if (device_count <= 0) {
            printf("DEBUG: 没有检测到 CUDA 设备\n");
            return false;
        }
        return true;
    }

    /**
     * 包装 Pinned Memory 分配
     */
    void* cuda_alloc_pinned(size_t size)
    {
        void* ptr = NULL;
        cudaError_t cuda_err = cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined);
        
        if (cuda_err != cudaSuccess) {
            printf("CUDA Pinned Alloc failed: %s", cudaGetErrorString(cuda_err));
            return NULL;
        }
        
        return ptr;
    }

    /**
     * 包装 Pinned Memory 释放
     */
    void cuda_free_pinned(void* ptr)
    {
        cudaError_t cuda_err = cudaFreeHost(ptr);
        if (cuda_err != cudaSuccess){
            printf("CUDA Pinned Free failed: %s\n", cudaGetErrorString(cuda_err));
        }
        return;
    }

    /**
     * 包装 GPU 内存分配
     */
    void** cuda_malloc(void** d_ptr, size_t size)
    {
        cudaError_t err = cudaMalloc((void**)d_ptr, size);
        if (err != cudaSuccess) {
            printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        }
        return d_ptr;
    }

    void cuda_free(void* d_ptr)
    {
        cudaError_t err = cudaFree(d_ptr);
        if (err != cudaSuccess) {
            printf("CUDA free failed: %s\n", cudaGetErrorString(err));
        }
    }

    /**
     * 包装 Host to Device 内存复制
     */
    void cuda_memcpy_h2d(void* d_dst, const void* h_src, size_t size)
    {
        cudaError_t err = cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA memcpy H2D failed: %s\n", cudaGetErrorString(err));
        }
    }

    /**
     * 包装 Device to Host 内存复制
     */
    void cuda_memcpy_d2h(void* h_dst, const void* d_src, size_t size)
    {
        
        cudaError_t err = cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("CUDA memcpy D2H failed: %s\n", cudaGetErrorString(err));
        }
    }

    /**
     * 包装 Host to Device 异步内存复制
     */
    void cuda_memcpy_async_h2d(void* d_dst, const void* h_src, size_t size)
    {
        cudaError_t err = cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA memcpy async H2D failed: %s\n", cudaGetErrorString(err));
        }
    }

    /**
     * 包装 Device to Host 异步内存复制
     */
    void cuda_memcpy_async_d2h(void* h_dst, const void* d_src, size_t size)
    {
        cudaError_t err = cudaMemcpyAsync(h_dst, d_src, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("CUDA memcpy async D2H failed: %s\n", cudaGetErrorString(err));
        }
    }

    /* ========================================================================= */
    /*                              Pipeline 管理函数                             */
    /* ========================================================================= */

    void cuda_pipeline_init(GpuPipelineContext* ctx, int dim, size_t total_vectors, 
                float* d_vec_ptr)
    {   
        if (!ctx) return;

        ctx->dimensions = dim;
        ctx->chunk_capacity = PIPELINE_CHUNK_SIZE;
        ctx->total_uploaded = 0;
        ctx->active_buf_idx = 0;
        ctx->d_vectors_base = d_vec_ptr;
        
        /* 初始化双缓冲资源 */
        for (int i = 0; i < 2; i++) {
            size_t bytes = ctx->chunk_capacity * dim * sizeof(float);
            
            /* 1. 分配 Pinned Memory */
            cudaError_t err = cudaHostAlloc((void**)&ctx->h_vec_buffers[i], bytes, cudaHostAllocWriteCombined);
            if (err != cudaSuccess) {
                printf("CUDA Pipeline Init: Alloc failed for buf %d: %s\n", i, cudaGetErrorString(err));
                ctx->h_vec_buffers[i] = NULL; /* 标记失败 */
            }
            
            ctx->current_counts[i] = 0;
            
            /* 2. 创建 CUDA Events (禁用计时以提高性能) */
            cudaEvent_t evt;
            cudaEventCreateWithFlags(&evt, cudaEventDisableTiming);
            ctx->events[i] = (void*)evt;
            
            /* 3. 预先记录 Event，防止第一次 Wait 时卡住或报错 */
            cudaEventRecord(evt, 0); 
        }
    }

    /* 
     * 通用 flush 函数（兼容旧接口）
     * 目前与 cuda_pipeline_flush_vectors_only 功能相同
     */
    void cuda_pipeline_flush(GpuPipelineContext* ctx)
    {
        cuda_pipeline_flush_vectors_only(ctx);
    }

    /* 
     * 双缓冲核心逻辑：
     * 1. 发起当前 Buffer 的异步传输
     * 2. 记录当前 Buffer 的完成事件
     * 3. 切换到下一个 Buffer
     * 4. 等待下一个 Buffer 的事件（确保 GPU 已经用完了它，CPU 才能覆盖）
     */
    void cuda_pipeline_flush_vectors_only(GpuPipelineContext* ctx)
    {
        if (!ctx) return;
        
        int idx = ctx->active_buf_idx;
        
        /* 如果当前 Buffer 是空的，直接返回，不需要 flush */
        if (ctx->current_counts[idx] == 0) return;
        
        size_t bytes_vec = ctx->current_counts[idx] * ctx->dimensions * sizeof(float);
        
        /* 计算 GPU 显存目标偏移 */
        float* d_dest = ctx->d_vectors_base + (ctx->total_uploaded * ctx->dimensions);
        
        /* --- 步骤 1: 异步 DMA 传输 (H2D) --- */
        /* 使用默认流 (0)，或者你可以传入一个专用流 */
        cudaMemcpyAsync(d_dest, ctx->h_vec_buffers[idx], bytes_vec, cudaMemcpyHostToDevice, 0);
        
        /* --- 步骤 2: 记录 Event --- */
        /* 告诉 GPU: "当你搞定上面那个 Memcpy 后，把这个 Event 设为 Signaled" */
        cudaEventRecord((cudaEvent_t)ctx->events[idx], 0);
        
        /* 更新统计 */
        ctx->total_uploaded += ctx->current_counts[idx];
        ctx->current_counts[idx] = 0; /* 清空计数，准备下一次复用 */
        
        /* --- 步骤 3: 切换索引 (Ping-Pong) --- */
        int next_idx = 1 - idx;
        
        /* --- 步骤 4: 等待下一个 Buffer 可用 --- */
        /* CPU 必须等待 GPU 完成对 next_idx Buffer 的操作（也就是上上一轮的传输） */
        /* 如果 GPU 跑得快，这里立即返回；如果 GPU 慢，CPU 会在这里阻塞，防止覆盖数据 */
        cudaEventSynchronize((cudaEvent_t)ctx->events[next_idx]);
        
        ctx->active_buf_idx = next_idx;
    }

    void cuda_pipeline_free(GpuPipelineContext* ctx)
    {
        if (!ctx) return;
        
        /* 确保所有 GPU 任务完成 */
        cudaDeviceSynchronize();

        for (int i = 0; i < 2; i++) {
            if (ctx->h_vec_buffers[i]) {
                cudaFreeHost(ctx->h_vec_buffers[i]);
                ctx->h_vec_buffers[i] = NULL;
            }
            if (ctx->events[i]) {
                cudaEventDestroy((cudaEvent_t)ctx->events[i]);
                ctx->events[i] = NULL;
            }
        }
    }

    /* 辅助清理函数 */
    void
    cuda_cleanup_memory(float* d_query_batch, int* d_cluster_size, float* d_cluster_vectors,
                    float* d_cluster_centers, int* d_initial_indices, float* d_topk_dist, int* d_topk_index)
    {
        if (d_query_batch) cudaFree(d_query_batch);
        if (d_cluster_size) cudaFree(d_cluster_size);
        if (d_cluster_vectors) cudaFree(d_cluster_vectors);
        if (d_cluster_centers) cudaFree(d_cluster_centers);
        if (d_initial_indices) cudaFree(d_initial_indices);
        if (d_topk_dist) cudaFree(d_topk_dist);
        if (d_topk_index) cudaFree(d_topk_index);
    }

    /* CUDA 错误检查函数 */
    void cuda_device_synchronize(void)
    {
        cudaDeviceSynchronize();
    }

    /* 获取最后一个 CUDA 错误的字符串描述 */
    const char* cuda_get_last_error_string(void)
    {
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) {
            return "no error";
        }
        return cudaGetErrorString(err);
    }

    /* 检查最后一个 CUDA 错误，如果有错误返回 true */
    bool cuda_check_last_error(void)
    {
        cudaError_t err = cudaGetLastError();
        return (err != cudaSuccess);
    }
}