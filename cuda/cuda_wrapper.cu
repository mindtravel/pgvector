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
        if (err != cudaSuccess) {
            printf("DEBUG: cudaGetDeviceCount 失败: %s\n", cudaGetErrorString(err));
            return false;
        }
        if (device_count <= 0) {
            printf("DEBUG: 没有检测到 CUDA 设备\n");
            return false;
        }
        printf("DEBUG: CUDA 可用，检测到 %d 个设备\n", device_count);
        return true;
    }

    
    // 简单的 CUDA 功能测试
    bool cuda_basic_test() {
        printf("DEBUG: 开始 CUDA 基本功能测试\n");
        
        // 测试设备设置
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            printf("ERROR: cudaSetDevice 失败: %s\n", cudaGetErrorString(err));
            return false;
        }
        printf("DEBUG: cudaSetDevice 成功\n");
        
        // 测试内存分配
        float *d_test;
        err = cudaMalloc(&d_test, 1024);
        if (err != cudaSuccess) {
            printf("ERROR: cudaMalloc 测试失败: %s\n", cudaGetErrorString(err));
            return false;
        }
        printf("DEBUG: cudaMalloc 测试成功\n");
        
        // 测试内存释放
        err = cudaFree(d_test);
        if (err != cudaSuccess) {
            printf("ERROR: cudaFree 测试失败: %s\n", cudaGetErrorString(err));
            return false;
        }
        printf("DEBUG: cudaFree 测试成功\n");
        
        printf("DEBUG: CUDA 基本功能测试通过\n");
        return true;
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

    // CUDA核函数：计算L2距离
    __global__ void compute_l2_distances_kernel(const float* centers, 
                                               const float* query_vector, 
                                               float* distances, 
                                               int num_centers, 
                                               int dimensions) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx < num_centers) {
            float sum = 0.0f;
            for (int i = 0; i < dimensions; i++) {
                float diff = centers[idx * dimensions + i] - query_vector[i];
                sum += diff * diff;
            }
            distances[idx] = sqrtf(sum);
        }
    }

    // CUDA核函数：计算余弦距离
    __global__ void compute_cosine_distances_kernel(const float* centers, 
                                                   const float* query_vector, 
                                                   float* distances, 
                                                   int num_centers, 
                                                   int dimensions) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx < num_centers) {
            float dot_product = 0.0f;
            float norm_query = 0.0f;
            float norm_center = 0.0f;
            
            for (int i = 0; i < dimensions; i++) {
                float q_val = query_vector[i];
                float c_val = centers[idx * dimensions + i];
                
                dot_product += q_val * c_val;
                norm_query += q_val * q_val;
                norm_center += c_val * c_val;
            }
            
            float norm_product = sqrtf(norm_query) * sqrtf(norm_center);
            if (norm_product > 0.0f) {
                distances[idx] = 1.0f - (dot_product / norm_product);
            } else {
                distances[idx] = 1.0f;
            }
        }
    }

    // 初始化GPU聚类中心搜索上下文
    CudaCenterSearchContext* cuda_center_search_init(int num_centers, int dimensions, bool use_zero_copy) {
        printf("DEBUG: 开始初始化 GPU 上下文\n");
        printf("DEBUG: 参数 - 聚类中心数: %d, 维度: %d, 零拷贝: %s\n", 
               num_centers, dimensions, use_zero_copy ? "是" : "否");
        
        if (!cuda_is_available()) {
            printf("ERROR: CUDA不可用，无法初始化GPU上下文\n");
            return NULL;
        }
        
        printf("DEBUG: CUDA 可用性检查通过\n");
        
        // 检查 CUDA 设备信息
        int device_count;
        cudaError_t device_err = cudaGetDeviceCount(&device_count);
        if (device_err == cudaSuccess) {
            printf("DEBUG: 检测到 %d 个 CUDA 设备\n", device_count);
            
            cudaDeviceProp prop;
            cudaError_t prop_err = cudaGetDeviceProperties(&prop, 0);
            if (prop_err == cudaSuccess) {
                printf("DEBUG: 设备 0: %s, 计算能力: %d.%d\n", 
                       prop.name, prop.major, prop.minor);
                printf("DEBUG: 总内存: %.2f GB, 可用内存: %.2f GB\n", 
                       prop.totalGlobalMem / (1024.0*1024.0*1024.0),
                       prop.totalGlobalMem / (1024.0*1024.0*1024.0));
            } else {
                printf("WARNING: 无法获取设备属性: %s\n", cudaGetErrorString(prop_err));
            }
        } else {
            printf("WARNING: 无法获取设备数量: %s\n", cudaGetErrorString(device_err));
        }

        // 检查参数有效性
        if (num_centers <= 0 || dimensions <= 0) {
            printf("ERROR: 无效参数 - 聚类中心数: %d, 维度: %d\n", num_centers, dimensions);
            return NULL;
        }

        CudaCenterSearchContext* ctx = (CudaCenterSearchContext*)malloc(sizeof(CudaCenterSearchContext));
        if (!ctx) {
            printf("ERROR: 无法分配CUDA上下文内存\n");
            return NULL;
        }

        memset(ctx, 0, sizeof(CudaCenterSearchContext));
        ctx->num_centers = num_centers;
        ctx->dimensions = dimensions;
        ctx->use_zero_copy = use_zero_copy;
        ctx->max_batch_size = 1024;  // 设置最大批量大小
        ctx->batch_support = true;   // 启用批量处理支持

        // 计算内存需求
        size_t centers_size = num_centers * dimensions * sizeof(float);
        size_t query_size = dimensions * sizeof(float);
        size_t distances_size = num_centers * sizeof(float);
        
        printf("INFO: 尝试分配GPU内存 - 聚类中心: %zu字节, 查询向量: %zu字节, 距离: %zu字节\n", 
               centers_size, query_size, distances_size);

        // 检查GPU内存是否足够
        size_t free_memory, total_memory;
        cudaError_t mem_err = cudaMemGetInfo(&free_memory, &total_memory);
        if (mem_err == cudaSuccess) {
            size_t required_memory = centers_size + query_size + distances_size;
            if (use_zero_copy) {
                required_memory += centers_size; // 零拷贝需要额外的页锁定内存
            }
            
            if (free_memory < required_memory) {
                printf("WARNING: GPU内存不足 - 需要: %zu字节, 可用: %zu字节\n", required_memory, free_memory);
                // 不直接返回NULL，尝试分配看看是否真的会失败
            }
        }

        cudaError_t err;
        
        if (use_zero_copy) {
            // 零拷贝模式：分配页锁定主机内存
            printf("INFO: 尝试分配页锁定内存 (%zu字节)\n", centers_size);
            err = cudaHostAlloc(&ctx->h_centers_pinned, centers_size, cudaHostAllocMapped);
            if (err != cudaSuccess) {
                printf("ERROR: 页锁定内存分配失败: %s (需要%zu字节)\n", 
                       cudaGetErrorString(err), centers_size);
                free(ctx);
                return NULL;
            }
            
            // 获取GPU可访问的地址
            err = cudaHostGetDevicePointer(&ctx->d_centers, ctx->h_centers_pinned, 0);
            if (err != cudaSuccess) {
                printf("ERROR: 获取GPU设备指针失败: %s\n", cudaGetErrorString(err));
                cudaFreeHost(ctx->h_centers_pinned);
                free(ctx);
                return NULL;
            }
            printf("INFO: 页锁定内存分配成功\n");
        } else {
            // 标准模式：分配GPU设备内存
            printf("INFO: 尝试分配GPU设备内存 (%zu字节)\n", centers_size);
            err = cudaMalloc(&ctx->d_centers, centers_size);
            if (err != cudaSuccess) {
                printf("ERROR: GPU设备内存分配失败: %s (需要%zu字节)\n", 
                       cudaGetErrorString(err), centers_size);
                free(ctx);
                return NULL;
            }
            printf("INFO: GPU设备内存分配成功\n");
        }

        // 分配查询向量内存
        err = cudaMalloc(&ctx->d_query_vector, query_size);
        if (err != cudaSuccess) {
            printf("ERROR: 查询向量内存分配失败: %s\n", cudaGetErrorString(err));
            if (use_zero_copy) {
                cudaFreeHost(ctx->h_centers_pinned);
            } else {
                cudaFree(ctx->d_centers);
            }
            free(ctx);
            return NULL;
        }

        // 分配距离结果内存
        err = cudaMalloc(&ctx->d_distances, distances_size);
        if (err != cudaSuccess) {
            printf("ERROR: 距离结果内存分配失败: %s\n", cudaGetErrorString(err));
            if (use_zero_copy) {
                cudaFreeHost(ctx->h_centers_pinned);
            } else {
                cudaFree(ctx->d_centers);
            }
            cudaFree(ctx->d_query_vector);
            free(ctx);
            return NULL;
        }

        // 分配批量查询向量内存
        size_t batch_query_size = ctx->max_batch_size * dimensions * sizeof(float);
        err = cudaMalloc(&ctx->d_batch_queries, batch_query_size);
        if (err != cudaSuccess) {
            printf("ERROR: 批量查询向量内存分配失败: %s\n", cudaGetErrorString(err));
            if (use_zero_copy) {
                cudaFreeHost(ctx->h_centers_pinned);
            } else {
                cudaFree(ctx->d_centers);
            }
            cudaFree(ctx->d_query_vector);
            cudaFree(ctx->d_distances);
            free(ctx);
            return NULL;
        }

        // 分配批量距离结果内存
        size_t batch_distances_size = ctx->max_batch_size * num_centers * sizeof(float);
        err = cudaMalloc(&ctx->d_batch_distances, batch_distances_size);
        if (err != cudaSuccess) {
            printf("ERROR: 批量距离结果内存分配失败: %s\n", cudaGetErrorString(err));
            if (use_zero_copy) {
                cudaFreeHost(ctx->h_centers_pinned);
            } else {
                cudaFree(ctx->d_centers);
            }
            cudaFree(ctx->d_query_vector);
            cudaFree(ctx->d_distances);
            cudaFree(ctx->d_batch_queries);
            free(ctx);
            return NULL;
        }

        ctx->initialized = true;
        printf("INFO: GPU上下文初始化成功\n");
        return ctx;
    }

    // 清理GPU聚类中心搜索上下文
    void cuda_center_search_cleanup(CudaCenterSearchContext* ctx) {
        if (!ctx) {
            printf("WARNING: 尝试清理空的CUDA上下文\n");
            return;
        }

        printf("INFO: 开始清理GPU上下文\n");

        if (ctx->initialized) {
            if (ctx->use_zero_copy && ctx->h_centers_pinned) {
                cudaError_t err = cudaFreeHost(ctx->h_centers_pinned);
                if (err != cudaSuccess) {
                    printf("WARNING: 释放页锁定内存失败: %s\n", cudaGetErrorString(err));
                } else {
                    printf("INFO: 页锁定内存释放成功\n");
                }
                ctx->h_centers_pinned = NULL;
            } else if (ctx->d_centers) {
                cudaError_t err = cudaFree(ctx->d_centers);
                if (err != cudaSuccess) {
                    printf("WARNING: 释放GPU设备内存失败: %s\n", cudaGetErrorString(err));
                } else {
                    printf("INFO: GPU设备内存释放成功\n");
                }
                ctx->d_centers = NULL;
            }
            
            if (ctx->d_query_vector) {
                cudaError_t err = cudaFree(ctx->d_query_vector);
                if (err != cudaSuccess) {
                    printf("WARNING: 释放查询向量内存失败: %s\n", cudaGetErrorString(err));
                } else {
                    printf("INFO: 查询向量内存释放成功\n");
                }
                ctx->d_query_vector = NULL;
            }
            
            if (ctx->d_distances) {
                cudaError_t err = cudaFree(ctx->d_distances);
                if (err != cudaSuccess) {
                    printf("WARNING: 释放距离结果内存失败: %s\n", cudaGetErrorString(err));
                } else {
                    printf("INFO: 距离结果内存释放成功\n");
                }
                ctx->d_distances = NULL;
            }
        }

        free(ctx);
        printf("INFO: GPU上下文清理完成\n");
    }

    // 上传聚类中心数据到GPU
    int cuda_upload_centers(CudaCenterSearchContext* ctx, const float* centers_data) {
        if (!ctx || !ctx->initialized || !centers_data) {
            printf("ERROR: 无效参数 - ctx: %p, initialized: %d, centers_data: %p\n", 
                   ctx, ctx ? ctx->initialized : 0, centers_data);
            return -1;
        }

        size_t size = ctx->num_centers * ctx->dimensions * sizeof(float);
        printf("INFO: 开始上传聚类中心数据到GPU (%zu字节, %d个中心, %d维)\n", 
               size, ctx->num_centers, ctx->dimensions);
        
        // 打印聚类中心数据内容用于调试
        printf("DEBUG: 聚类中心数据内容:\n");
        for (int i = 0; i < ctx->num_centers; i++) {
            printf("DEBUG: 聚类中心%d前5个元素:\n", i);
            for (int j = 0; j < (ctx->dimensions < 5 ? ctx->dimensions : 5); j++) {
                int idx = i * ctx->dimensions + j;
                printf("DEBUG:   centers_data[%d] = %f\n", idx, centers_data[idx]);
            }
        }
        
        cudaError_t err = cudaMemcpy(ctx->d_centers, centers_data, size, cudaMemcpyHostToDevice);
        
        if (err != cudaSuccess) {
            printf("ERROR: 聚类中心数据上传失败: %s\n", cudaGetErrorString(err));
            return -1;
        }

        printf("INFO: 聚类中心数据上传成功\n");
        return 0;
    }

    // 计算聚类中心距离
    int cuda_compute_center_distances(CudaCenterSearchContext* ctx, 
                                     const float* query_vector, 
                                     float* distances) {
        if (!ctx || !ctx->initialized || !query_vector || !distances) {
            return -1;
        }

        // 上传查询向量到GPU
        size_t query_size = ctx->dimensions * sizeof(float);
        cudaError_t err = cudaMemcpy(ctx->d_query_vector, query_vector, query_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return -1;
        }

        // 设置CUDA核函数参数
        int block_size = 256;
        int grid_size = (ctx->num_centers + block_size - 1) / block_size;

        // 调用L2距离计算核函数
        compute_l2_distances_kernel<<<grid_size, block_size>>>(
            ctx->d_centers, 
            ctx->d_query_vector, 
            ctx->d_distances, 
            ctx->num_centers, 
            ctx->dimensions
        );

        // 同步设备
        cudaDeviceSynchronize();

        // 检查错误
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return -1;
        }

        // 将结果复制回主机
        size_t distances_size = ctx->num_centers * sizeof(float);
        err = cudaMemcpy(distances, ctx->d_distances, distances_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return -1;
        }

        return 0;
    }

    // 零拷贝方式上传聚类中心数据
    int cuda_upload_centers_zero_copy(CudaCenterSearchContext* ctx, const float* centers_data) {
        if (!ctx || !ctx->initialized || !centers_data || !ctx->use_zero_copy) {
            printf("ERROR: 零拷贝上传参数无效 - ctx: %p, initialized: %d, centers_data: %p, use_zero_copy: %d\n", 
                   ctx, ctx ? ctx->initialized : 0, centers_data, ctx ? ctx->use_zero_copy : 0);
            return -1;
        }

        if (!ctx->h_centers_pinned) {
            printf("ERROR: 页锁定内存指针为空\n");
            return -1;
        }

        size_t size = ctx->num_centers * ctx->dimensions * sizeof(float);
        printf("INFO: 开始零拷贝上传聚类中心数据 (%zu字节, %d个中心, %d维)\n", 
               size, ctx->num_centers, ctx->dimensions);
        
        // 直接复制到页锁定内存，GPU可以直接访问
        memcpy(ctx->h_centers_pinned, centers_data, size);
        
        printf("INFO: 零拷贝聚类中心数据上传成功\n");
        return 0;
    }

    // 设置零拷贝模式
    int cuda_set_zero_copy_mode(CudaCenterSearchContext* ctx, bool enable) {
        if (!ctx || !ctx->initialized) {
            return -1;
        }

        // 如果模式没有改变，直接返回
        if (ctx->use_zero_copy == enable) {
            return 0;
        }

        // 重新分配内存（简化实现，实际应用中可能需要更复杂的逻辑）
        ctx->use_zero_copy = enable;
        
        return 0;
    }

    // CUDA核函数：批量计算probes列表的L2距离
    __global__ void compute_batch_probes_l2_distances_kernel(const float* probes_data,
                                                           const int* probes_offsets,
                                                           const int* probes_counts,
                                                           const float* batch_queries,
                                                           float* batch_distances,
                                                           int num_probes_lists,
                                                           int num_queries,
                                                           int dimensions) {
        int probe_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (query_idx >= num_queries) return;
        
        // 计算当前probe在全局probes数组中的索引
        int global_probe_idx = 0;
        for (int list_idx = 0; list_idx < num_probes_lists; list_idx++) {
            int list_count = probes_counts[list_idx];
            if (probe_idx < list_count) {
                global_probe_idx = probes_offsets[list_idx] + probe_idx;
                break;
            }
            probe_idx -= list_count;
        }
        
        // 计算总probes向量数量
        int total_probes_vectors = probes_offsets[num_probes_lists - 1] + probes_counts[num_probes_lists - 1];
        
        if (global_probe_idx < total_probes_vectors) {
            float sum = 0.0f;
            for (int i = 0; i < dimensions; i++) {
                float diff = probes_data[global_probe_idx * dimensions + i] - 
                            batch_queries[query_idx * dimensions + i];
                sum += diff * diff;
            }
            // 存储结果：batch_distances[query_idx * total_probes + global_probe_idx]
            batch_distances[query_idx * total_probes_vectors + global_probe_idx] = sqrtf(sum);
        }
    }

    // CUDA核函数：批量计算probes列表的余弦距离
    __global__ void compute_batch_probes_cosine_distances_kernel(const float* probes_data,
                                                               const int* probes_offsets,
                                                               const int* probes_counts,
                                                               const float* batch_queries,
                                                               float* batch_distances,
                                                               int num_probes_lists,
                                                               int num_queries,
                                                               int dimensions) {
        int probe_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (query_idx >= num_queries) return;
        
        // 计算当前probe在全局probes数组中的索引
        int global_probe_idx = 0;
        for (int list_idx = 0; list_idx < num_probes_lists; list_idx++) {
            int list_count = probes_counts[list_idx];
            if (probe_idx < list_count) {
                global_probe_idx = probes_offsets[list_idx] + probe_idx;
                break;
            }
            probe_idx -= list_count;
        }
        
        // 计算总probes向量数量
        int total_probes_vectors = probes_offsets[num_probes_lists - 1] + probes_counts[num_probes_lists - 1];
        
        if (global_probe_idx < total_probes_vectors) {
            float dot_product = 0.0f;
            float norm_query = 0.0f;
            float norm_probe = 0.0f;
            
            for (int i = 0; i < dimensions; i++) {
                float q_val = batch_queries[query_idx * dimensions + i];
                float p_val = probes_data[global_probe_idx * dimensions + i];
                
                dot_product += q_val * p_val;
                norm_query += q_val * q_val;
                norm_probe += p_val * p_val;
            }
            
            float norm_product = sqrtf(norm_query) * sqrtf(norm_probe);
            if (norm_product > 0.0f) {
                batch_distances[query_idx * total_probes_vectors + global_probe_idx] = 1.0f - (dot_product / norm_product);
            } else {
                batch_distances[query_idx * total_probes_vectors + global_probe_idx] = 1.0f;
            }
        }
    }

    // 批量GPU距离计算核函数
    __global__ void compute_batch_l2_distances_kernel(const float* centers, 
                                                     const float* batch_queries, 
                                                     float* batch_distances, 
                                                     int num_centers, 
                                                     int num_queries,
                                                     int dimensions) {
        int center_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
        
        // 添加更严格的边界检查
        if (center_idx >= num_centers || query_idx >= num_queries) {
            return;
        }
        
        // 检查指针有效性
        if (!centers || !batch_queries || !batch_distances) {
            return;
        }
        
        float sum = 0.0f;
        for (int i = 0; i < dimensions; i++) {
            // 添加数组边界检查
            int center_offset = center_idx * dimensions + i;
            int query_offset = query_idx * dimensions + i;
            
            if (center_offset >= num_centers * dimensions || 
                query_offset >= num_queries * dimensions) {
                return;
            }
            
            float diff = centers[center_offset] - batch_queries[query_offset];
            sum += diff * diff;
        }
        
        // 存储结果：batch_distances[query_idx * num_centers + center_idx]
        int result_offset = query_idx * num_centers + center_idx;
        if (result_offset < num_queries * num_centers) {
            float distance = sqrtf(sum);
            batch_distances[result_offset] = distance;
            
            // 添加调试信息（只对前几个结果）
            if (query_idx == 0 && center_idx < 2) {
                printf("DEBUG: CUDA核函数 - 查询%d, 聚类中心%d, 距离=%.6f\n", 
                       query_idx, center_idx, distance);
            }
        }
    }

    // 批量GPU距离计算核函数（余弦距离）
    __global__ void compute_batch_cosine_distances_kernel(const float* centers, 
                                                         const float* batch_queries, 
                                                         float* batch_distances, 
                                                         int num_centers, 
                                                         int num_queries,
                                                         int dimensions) {
        int center_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (center_idx < num_centers && query_idx < num_queries) {
            float dot_product = 0.0f;
            float norm_query = 0.0f;
            float norm_center = 0.0f;
            
            for (int i = 0; i < dimensions; i++) {
                float q_val = batch_queries[query_idx * dimensions + i];
                float c_val = centers[center_idx * dimensions + i];
                
                dot_product += q_val * c_val;
                norm_query += q_val * q_val;
                norm_center += c_val * c_val;
            }
            
            float norm_product = sqrtf(norm_query) * sqrtf(norm_center);
            if (norm_product > 0.0f) {
                batch_distances[query_idx * num_centers + center_idx] = 1.0f - (dot_product / norm_product);
            } else {
                batch_distances[query_idx * num_centers + center_idx] = 1.0f;
            }
        }
    }

    // 批量聚类中心距离计算
    int cuda_compute_batch_center_distances(CudaCenterSearchContext* ctx,
                                           const float* batch_query_vectors,
                                           int num_queries,
                                           float* batch_distances) {
        printf("DEBUG: cuda_compute_batch_center_distances 开始执行\n");
        printf("DEBUG: 参数 - ctx: %p, initialized: %d, batch_query_vectors: %p, batch_distances: %p\n", 
               ctx, ctx ? ctx->initialized : 0, batch_query_vectors, batch_distances);
        printf("DEBUG: 查询数量: %d, 最大批量大小: %d\n", num_queries, ctx ? ctx->max_batch_size : 0);
        
        if (!ctx || !ctx->initialized || !batch_query_vectors || !batch_distances) {
            printf("ERROR: 参数检查失败\n");
            return -1;
        }

        if (!ctx->batch_support) {
            printf("ERROR: 批量处理不支持\n");
            return -1;
        }

        if (num_queries > ctx->max_batch_size) {
            printf("ERROR: 批量大小超出限制 - 查询数量: %d, 最大批量大小: %d\n", num_queries, ctx->max_batch_size);
            return -1;  // 批量大小超出限制
        }

        // 打印前几个查询向量的前几个元素用于调试
        printf("DEBUG: 批量查询向量数据内容:\n");
        int max_queries = (2 < num_queries) ? 2 : num_queries;
        for (int i = 0; i < max_queries; i++) {
            printf("DEBUG: 查询向量%d前5个元素:\n", i);
            int max_dims = (5 < ctx->dimensions) ? 5 : ctx->dimensions;
            for (int j = 0; j < max_dims; j++) {
                int idx = i * ctx->dimensions + j;
                printf("DEBUG:   batch_query_vectors[%d] = %f\n", idx, batch_query_vectors[idx]);
            }
        }

        // 上传批量查询向量到GPU
        size_t batch_size = num_queries * ctx->dimensions * sizeof(float);
        printf("DEBUG: 开始上传批量查询向量到GPU，大小: %zu字节\n", batch_size);
        cudaError_t err = cudaMemcpy(ctx->d_batch_queries, batch_query_vectors, 
                                    batch_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("ERROR: 批量查询向量上传失败: %s\n", cudaGetErrorString(err));
            return -1;
        }
        printf("DEBUG: 批量查询向量上传成功\n");

        // 设置CUDA核函数参数
        // 使用2D网格：x维度处理聚类中心，y维度处理查询向量
        dim3 block_size(16, 16);  // 16x16 = 256个线程
        dim3 grid_size((ctx->num_centers + block_size.x - 1) / block_size.x,
                      (num_queries + block_size.y - 1) / block_size.y);

        // 调用批量L2距离计算核函数
        printf("DEBUG: 调用CUDA核函数 - 网格大小: (%d,%d), 块大小: (%d,%d)\n", 
               grid_size.x, grid_size.y, block_size.x, block_size.y);
        compute_batch_l2_distances_kernel<<<grid_size, block_size>>>(
            ctx->d_centers, 
            ctx->d_batch_queries, 
            ctx->d_batch_distances, 
            ctx->num_centers, 
            num_queries,
            ctx->dimensions
        );

        // 同步设备
        printf("DEBUG: 同步CUDA设备\n");
        cudaDeviceSynchronize();

        // 检查错误
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ERROR: CUDA核函数执行失败: %s\n", cudaGetErrorString(err));
            return -1;
        }
        printf("DEBUG: CUDA核函数执行成功\n");

        // 将结果复制回主机
        size_t result_size = num_queries * ctx->num_centers * sizeof(float);
        err = cudaMemcpy(batch_distances, ctx->d_batch_distances, 
                        result_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return -1;
        }

        return 0;
    }

    // 批量余弦距离计算
    int cuda_compute_batch_cosine_distances(CudaCenterSearchContext* ctx,
                                           const float* batch_query_vectors,
                                           int num_queries,
                                           float* batch_distances) {
        if (!ctx || !ctx->initialized || !batch_query_vectors || !batch_distances) {
            return -1;
        }

        if (!ctx->batch_support) {
            return -1;
        }

        if (num_queries > ctx->max_batch_size) {
            return -1;  // 批量大小超出限制
        }

        // 上传批量查询向量到GPU
        size_t batch_size = num_queries * ctx->dimensions * sizeof(float);
        cudaError_t err = cudaMemcpy(ctx->d_batch_queries, batch_query_vectors, 
                                    batch_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return -1;
        }

        // 设置CUDA核函数参数
        dim3 block_size(16, 16);
        dim3 grid_size((ctx->num_centers + block_size.x - 1) / block_size.x,
                      (num_queries + block_size.y - 1) / block_size.y);

        // 调用批量余弦距离计算核函数
        compute_batch_cosine_distances_kernel<<<grid_size, block_size>>>(
            ctx->d_centers, 
            ctx->d_batch_queries, 
            ctx->d_batch_distances, 
            ctx->num_centers, 
            num_queries,
            ctx->dimensions
        );
        
        // 同步设备
        cudaDeviceSynchronize();
        
        // 检查错误
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return -1;
        }

        // 将结果复制回主机
        size_t result_size = num_queries * ctx->num_centers * sizeof(float);
        err = cudaMemcpy(batch_distances, ctx->d_batch_distances, 
                        result_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return -1;
        }

        return 0;
    }

    // 上传probes列表数据到GPU
    int cuda_upload_probes_data(CudaCenterSearchContext* ctx,
                               const float* probes_data,
                               const int* probes_offsets,
                               const int* probes_counts,
                               int num_probes_lists,
                               int total_probes_vectors,
                               int dimensions) {
        if (!ctx || !ctx->initialized || !probes_data || !probes_offsets || !probes_counts) {
            return -1;
        }

        // 如果已经上传过，先清理旧数据
        if (ctx->probes_uploaded) {
            if (ctx->d_probes_data) {
                cudaFree(ctx->d_probes_data);
                ctx->d_probes_data = NULL;
            }
            if (ctx->d_probes_offsets) {
                cudaFree(ctx->d_probes_offsets);
                ctx->d_probes_offsets = NULL;
            }
            if (ctx->d_probes_counts) {
                cudaFree(ctx->d_probes_counts);
                ctx->d_probes_counts = NULL;
            }
        }

        // 分配GPU内存
        size_t probes_data_size = total_probes_vectors * dimensions * sizeof(float);
        size_t offsets_size = num_probes_lists * sizeof(int);
        size_t counts_size = num_probes_lists * sizeof(int);

        cudaError_t err;

        // 分配probes向量数据内存
        err = cudaMalloc(&ctx->d_probes_data, probes_data_size);
        if (err != cudaSuccess) {
            return -1;
        }

        // 分配probes偏移量内存
        err = cudaMalloc(&ctx->d_probes_offsets, offsets_size);
        if (err != cudaSuccess) {
            cudaFree(ctx->d_probes_data);
            return -1;
        }

        // 分配probes计数内存
        err = cudaMalloc(&ctx->d_probes_counts, counts_size);
        if (err != cudaSuccess) {
            cudaFree(ctx->d_probes_data);
            cudaFree(ctx->d_probes_offsets);
            return -1;
        }

        // 上传数据到GPU
        err = cudaMemcpy(ctx->d_probes_data, probes_data, probes_data_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(ctx->d_probes_data);
            cudaFree(ctx->d_probes_offsets);
            cudaFree(ctx->d_probes_counts);
            return -1;
        }

        err = cudaMemcpy(ctx->d_probes_offsets, probes_offsets, offsets_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(ctx->d_probes_data);
            cudaFree(ctx->d_probes_offsets);
            cudaFree(ctx->d_probes_counts);
            return -1;
        }

        err = cudaMemcpy(ctx->d_probes_counts, probes_counts, counts_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(ctx->d_probes_data);
            cudaFree(ctx->d_probes_offsets);
            cudaFree(ctx->d_probes_counts);
            return -1;
        }
        
        // 更新上下文信息
        ctx->num_probes_lists = num_probes_lists;
        ctx->total_probes_vectors = total_probes_vectors;
        ctx->probes_uploaded = true;

        return 0;
    }
    
    // 批量probes距离计算
    int cuda_compute_batch_probes_distances(CudaCenterSearchContext* ctx,
                                           const float* batch_query_vectors,
                                           int num_queries,
                                           float* batch_distances) {
        if (!ctx || !ctx->initialized || !batch_query_vectors || !batch_distances) {
            return -1;
        }

        if (!ctx->probes_uploaded) {
            return -1;  // probes数据未上传
        }

        if (!ctx->batch_support) {
            return -1;
        }

        if (num_queries > ctx->max_batch_size) {
            return -1;  // 批量大小超出限制
        }

        // 上传批量查询向量到GPU
        size_t batch_size = num_queries * ctx->dimensions * sizeof(float);
        cudaError_t err = cudaMemcpy(ctx->d_batch_queries, batch_query_vectors, 
                                    batch_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return -1;
        }

        // 设置CUDA核函数参数
        // 使用2D网格：x维度处理probes向量，y维度处理查询向量
        dim3 block_size(16, 16);  // 16x16 = 256个线程
        dim3 grid_size((ctx->total_probes_vectors + block_size.x - 1) / block_size.x,
                      (num_queries + block_size.y - 1) / block_size.y);

        // 调用批量probes L2距离计算核函数
        compute_batch_probes_l2_distances_kernel<<<grid_size, block_size>>>(
            ctx->d_probes_data,
            ctx->d_probes_offsets,
            ctx->d_probes_counts,
            ctx->d_batch_queries,
            ctx->d_batch_distances,
            ctx->num_probes_lists,
            num_queries,
            ctx->dimensions
        );

        // 同步设备
        cudaDeviceSynchronize();

        // 检查错误
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return -1;
        }

        // 将结果复制回主机
        size_t result_size = num_queries * ctx->total_probes_vectors * sizeof(float);
        err = cudaMemcpy(batch_distances, ctx->d_batch_distances, 
                        result_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return -1;
        }

        return 0;
    }
    
}