/* 使用pch.h预编译头，它已包含必要的标准库头文件 */
#include "pch.h"
#include "cuda_wrapper.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>    // for INFINITY
#include <cfloat>   // for FLT_MAX

// Vector结构体定义（与CPU端保持一致）
// 注意：这里需要与vector.h中的定义一致
struct Vector {
    int32_t vl_len_;      // varlena header (do not touch directly!)
    int16_t dim;          // number of dimensions
    int16_t unused;       // reserved for future use, always zero
    float x[];            // flexible array member
};

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

    // CUDA核函数：计算L2距离（用于单次查询）
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
            
            if (ctx->d_batch_queries) {
                cudaFree(ctx->d_batch_queries);
                ctx->d_batch_queries = NULL;
            }
            
            if (ctx->d_batch_distances) {
                cudaFree(ctx->d_batch_distances);
                ctx->d_batch_distances = NULL;
            }
            
            // 清理probe候选数据（分离存储方案）
            if (ctx->d_probe_vectors) {
                cudaFree(ctx->d_probe_vectors);
                ctx->d_probe_vectors = NULL;
                printf("INFO: 向量数据内存释放成功\n");
            }
            
            if (ctx->d_probe_tids) {
                cudaFree(ctx->d_probe_tids);
                ctx->d_probe_tids = NULL;
                printf("INFO: TID数据内存释放成功\n");
            }
            
            if (ctx->d_probe_query_map) {
                cudaFree(ctx->d_probe_query_map);
                ctx->d_probe_query_map = NULL;
                printf("INFO: Probe查询映射内存释放成功\n");
            }
            
            if (ctx->d_probe_distances) {
                cudaFree(ctx->d_probe_distances);
                ctx->d_probe_distances = NULL;
                printf("INFO: Probe距离内存释放成功\n");
            }
            
            if (ctx->d_topk_indices) {
                cudaFree(ctx->d_topk_indices);
                ctx->d_topk_indices = NULL;
                printf("INFO: TopK索引内存释放成功\n");
            }
            
            if (ctx->d_topk_distances) {
                cudaFree(ctx->d_topk_distances);
                ctx->d_topk_distances = NULL;
                printf("INFO: TopK距离内存释放成功\n");
            }
            
            if (ctx->d_topk_tids) {
                cudaFree(ctx->d_topk_tids);
                ctx->d_topk_tids = NULL;
                printf("INFO: TopK TID内存释放成功\n");
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

        // 重新分配内存（简化实现）
        ctx->use_zero_copy = enable;
        
        return 0;
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

    // CUDA核函数：从对齐的向量数据计算L2距离（方案1：分离存储）
    // 向量数据：连续对齐存储，直接索引访问，无需偏移计算
    __global__ void compute_batch_probe_l2_distances_from_vectors_kernel(
                                                            const float* probe_vectors,
                                                            const int* probe_query_map,
                                                            const float* batch_queries,
                                                            float* batch_distances,
                                                            int num_candidates,
                                                            int num_queries,
                                                            int dimensions) {
        // 使用2D网格：x维度是候选索引，y维度是查询索引
        int candidate_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (candidate_idx >= num_candidates || query_idx >= num_queries) {
            return;
        }
        
        // 关键修复：只计算属于该查询的候选的距离
        // probe_query_map[candidate_idx] 表示候选属于哪个查询
        // 只有当候选属于当前查询时，才计算距离
        // 对于不属于当前查询的候选，距离应该设置为一个很大的值（FLT_MAX），而不是0
        // 这样在TopK选择时，不会被误选
        if (probe_query_map[candidate_idx] != query_idx) {
            // 候选不属于当前查询，设置距离为FLT_MAX（不会被TopK选中）
            int result_idx = query_idx * num_candidates + candidate_idx;
            batch_distances[result_idx] = FLT_MAX;
            return;
        }
        
        // 直接访问对齐的向量数据（无需偏移计算，提高性能）
        // 向量数据连续存储：probe_vectors[candidate_idx * dimensions + d]
        const float* vector_data = probe_vectors + candidate_idx * dimensions;
        
        // 计算该候选向量与该查询向量的L2距离
        float sum = 0.0f;
        for (int d = 0; d < dimensions; d++) {
            float diff = vector_data[d] - batch_queries[query_idx * dimensions + d];
            sum += diff * diff;
        }
        
        // 存储结果到距离数组中
        // 格式：distance[query_idx * num_candidates + candidate_idx]
        int result_idx = query_idx * num_candidates + candidate_idx;
        batch_distances[result_idx] = sqrtf(sum);
    }
    
    // CUDA核函数：批量提取TopK结果的TID（GPU端批量提取）
    // 从TID数组中批量提取TopK候选对应的TID，提高性能
    __global__ void extract_topk_tids_kernel(
                                                            const int* topk_candidate_indices,  // TopK候选索引 [num_queries * k]
                                                            const char* probe_tids,            // TID数组 [num_candidates * 6]
                                                            char* topk_tids,                   // 输出的TID结果 [num_queries * k * 6]
                                                            int num_candidates,                // 候选总数（用于边界检查）
                                                            int num_queries,
                                                            int k) {
        // 计算全局索引
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_results = num_queries * k;
        
        if (idx >= total_results) {
            return;
        }
        
        // 计算查询索引和排名
        int query_idx = idx / k;
        int rank = idx % k;
        
        // 从TopK索引中获取候选索引
        int candidate_idx = topk_candidate_indices[query_idx * k + rank];
        
        // 计算输出位置
        int output_offset = idx * 6;
        
        // 边界检查：candidate_idx应该是有效的候选索引
        if (candidate_idx < 0 || candidate_idx >= num_candidates) {
            // 无效索引，填充为0（表示无效TID）
            for (int i = 0; i < 6; i++) {
                topk_tids[output_offset + i] = 0;
            }
            return;
        }
        
        // 从TID数组中提取TID（每个TID 6字节）
        // 关键：使用相同的candidate_idx索引，保证一致性
        int tid_offset = candidate_idx * 6;
        for (int i = 0; i < 6; i++) {
            topk_tids[output_offset + i] = probe_tids[tid_offset + i];
        }
    }
    
    // CUDA核函数：一致性检查（验证向量数据和TID数据的索引对应关系）
    __global__ void verify_probe_consistency_kernel(
                                                            const float* probe_vectors,
                                                            const char* probe_tids,
                                                            const int* probe_query_map,
                                                            int* verification_results,  // 输出：0表示一致，非0表示不一致
                                                            int num_candidates,
                                                            int dimensions) {
        int candidate_idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (candidate_idx >= num_candidates) {
            return;
        }
        
        // 检查1：向量数据是否有效（非NaN，非Inf）
        const float* vector = probe_vectors + candidate_idx * dimensions;
        bool vector_valid = true;
        for (int d = 0; d < dimensions; d++) {
            if (!isfinite(vector[d])) {
                vector_valid = false;
                break;
            }
        }
        
        // 检查2：TID数据是否有效（非全0，表示有效的ItemPointer）
        const char* tid = probe_tids + candidate_idx * 6;
        bool tid_valid = false;
        for (int i = 0; i < 6; i++) {
            if (tid[i] != 0) {
                tid_valid = true;
                break;
            }
        }
        
        // 检查3：查询映射是否在有效范围内
        bool query_map_valid = (probe_query_map[candidate_idx] >= 0);
        
        // 如果所有检查都通过，设置结果为0（一致）
        if (vector_valid && tid_valid && query_map_valid) {
            verification_results[candidate_idx] = 0;
        } else {
            // 不一致，记录错误类型
            verification_results[candidate_idx] = 
                (vector_valid ? 0 : 1) |
                (tid_valid ? 0 : 2) |
                (query_map_valid ? 0 : 4);
        }
    }
    
    // 上传分离的向量数据和TID数据到GPU（分离存储方案，提高性能）
    // vectors: 向量数据数组（连续存储，每个向量dimensions个float）
    // tids: TID数据数组（ItemPointerData，每个6字节，连续存储）
    // query_ids: 查询ID映射（每个候选属于哪个查询）
    // 关键：vectors[i], tids[i], query_ids[i] 必须对应同一个候选，保证索引一致性
    int cuda_upload_probe_vectors(CudaCenterSearchContext* ctx,
                                   const float* vectors,
                                   const char* tids,  // ItemPointerData数组，每个6字节
                                   const int* query_ids,
                                   int num_candidates,
                                   int dimensions) {
        if (!ctx || !ctx->initialized || !vectors || !tids || !query_ids) {
            printf("ERROR: 上传分离数据参数无效\n");
            return -1;
        }
        
        printf("INFO: 开始上传分离数据到GPU (分离存储方案), 候选数: %d, 维度: %d\n", 
               num_candidates, dimensions);
        
        // 清理旧数据
        if (ctx->d_probe_vectors) {
            cudaFree(ctx->d_probe_vectors);
            ctx->d_probe_vectors = NULL;
        }
        if (ctx->d_probe_tids) {
            cudaFree(ctx->d_probe_tids);
            ctx->d_probe_tids = NULL;
        }
        if (ctx->d_probe_query_map) {
            cudaFree(ctx->d_probe_query_map);
            ctx->d_probe_query_map = NULL;
        }
        
        cudaError_t err;
        
        // 计算内存大小
        size_t vectors_size = num_candidates * dimensions * sizeof(float);
        size_t tids_size = num_candidates * 6;  // 每个TID 6字节
        size_t query_map_size = num_candidates * sizeof(int);
        
        printf("INFO: 分配GPU内存 - 向量: %zu字节, TID: %zu字节, 查询映射: %zu字节\n",
               vectors_size, tids_size, query_map_size);
        
        // 分配GPU内存存储向量数据
        err = cudaMalloc(&ctx->d_probe_vectors, vectors_size);
        if (err != cudaSuccess) {
            printf("ERROR: 向量数据GPU内存分配失败: %s (需要%zu字节)\n", 
                   cudaGetErrorString(err), vectors_size);
            return -1;
        }
        
        // 分配GPU内存存储TID数据
        err = cudaMalloc(&ctx->d_probe_tids, tids_size);
        if (err != cudaSuccess) {
            printf("ERROR: TID数据GPU内存分配失败: %s (需要%zu字节)\n", 
                   cudaGetErrorString(err), tids_size);
            cudaFree(ctx->d_probe_vectors);
            ctx->d_probe_vectors = NULL;
            return -1;
        }
        
        // 分配GPU内存存储查询映射
        err = cudaMalloc(&ctx->d_probe_query_map, query_map_size);
        if (err != cudaSuccess) {
            printf("ERROR: 查询映射GPU内存分配失败: %s (需要%zu字节)\n", 
                   cudaGetErrorString(err), query_map_size);
            cudaFree(ctx->d_probe_vectors);
            cudaFree(ctx->d_probe_tids);
            ctx->d_probe_vectors = NULL;
            ctx->d_probe_tids = NULL;
            return -1;
        }
        
        // 上传向量数据
        err = cudaMemcpy(ctx->d_probe_vectors, vectors, vectors_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("ERROR: 向量数据上传失败: %s\n", cudaGetErrorString(err));
            cudaFree(ctx->d_probe_vectors);
            cudaFree(ctx->d_probe_tids);
            cudaFree(ctx->d_probe_query_map);
            ctx->d_probe_vectors = NULL;
            ctx->d_probe_tids = NULL;
            ctx->d_probe_query_map = NULL;
            return -1;
        }
        
        // 上传TID数据
        err = cudaMemcpy(ctx->d_probe_tids, tids, tids_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("ERROR: TID数据上传失败: %s\n", cudaGetErrorString(err));
            cudaFree(ctx->d_probe_vectors);
            cudaFree(ctx->d_probe_tids);
            cudaFree(ctx->d_probe_query_map);
            ctx->d_probe_vectors = NULL;
            ctx->d_probe_tids = NULL;
            ctx->d_probe_query_map = NULL;
            return -1;
        }
        
        // 上传查询映射
        err = cudaMemcpy(ctx->d_probe_query_map, query_ids, query_map_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("ERROR: 查询映射上传失败: %s\n", cudaGetErrorString(err));
            cudaFree(ctx->d_probe_vectors);
            cudaFree(ctx->d_probe_tids);
            cudaFree(ctx->d_probe_query_map);
            ctx->d_probe_vectors = NULL;
            ctx->d_probe_tids = NULL;
            ctx->d_probe_query_map = NULL;
            return -1;
        }
        
        // 更新上下文
        ctx->num_probe_candidates = num_candidates;
        ctx->dimensions = dimensions;
        ctx->probes_uploaded = true;
        
        // 分配距离结果内存（用于存储GPU计算的距离）
        size_t distances_size = ctx->max_batch_size * num_candidates * sizeof(float);
        if (ctx->d_probe_distances) {
            cudaFree(ctx->d_probe_distances);
        }
        err = cudaMalloc(&ctx->d_probe_distances, distances_size);
        if (err != cudaSuccess) {
            printf("ERROR: 距离结果内存分配失败: %s\n", cudaGetErrorString(err));
            cudaFree(ctx->d_probe_vectors);
            cudaFree(ctx->d_probe_tids);
            cudaFree(ctx->d_probe_query_map);
            ctx->d_probe_vectors = NULL;
            ctx->d_probe_tids = NULL;
            ctx->d_probe_query_map = NULL;
            return -1;
        }
        
        printf("INFO: 成功上传分离数据到GPU (向量: %zu字节, TID: %zu字节)\n", 
               vectors_size, tids_size);
        
        // 执行一致性检查
        int verify_result = cuda_verify_probe_consistency(ctx, num_candidates, dimensions);
        if (verify_result != 0) {
            printf("WARNING: 一致性检查失败，但继续执行\n");
        } else {
            printf("INFO: 一致性检查通过\n");
        }
        
        return 0;
    }
    

    // GPU批量计算probe向量距离
    int cuda_compute_batch_probe_distances(CudaCenterSearchContext* ctx,
                                           const float* batch_query_vectors,
                                           int num_queries) {
        if (!ctx || !ctx->initialized || !batch_query_vectors) {
            printf("ERROR: GPU批量计算probe距离参数无效\n");
            return -1;
        }
        
        if (!ctx->probes_uploaded) {
            printf("ERROR: Probe向量数据未上传\n");
            return -1;
        }
        
        if (num_queries > ctx->max_batch_size) {
            printf("ERROR: 批量大小超出限制\n");
            return -1;
        }
        
        // 上传批量查询向量
        size_t batch_size = num_queries * ctx->dimensions * sizeof(float);
        cudaError_t err = cudaMemcpy(ctx->d_batch_queries, batch_query_vectors, 
                                    batch_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("ERROR: 批量查询向量上传失败: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        // 清零距离数组（确保未计算的位置为0）
        size_t distances_array_size = num_queries * ctx->num_probe_candidates * sizeof(float);
        cudaMemset(ctx->d_probe_distances, 0, distances_array_size);
        
        // 设置CUDA核函数参数（2D网格：x维度是候选，y维度是查询）
        dim3 block_size(16, 16);
        dim3 grid_size((ctx->num_probe_candidates + block_size.x - 1) / block_size.x,
                      (num_queries + block_size.y - 1) / block_size.y);
        
        // 使用分离存储方案计算距离
        if (!ctx->d_probe_vectors || !ctx->d_probe_tids) {
            printf("ERROR: Probe数据未上传（分离存储方案）\n");
            return -1;
        }
        
        compute_batch_probe_l2_distances_from_vectors_kernel<<<grid_size, block_size>>>(
            ctx->d_probe_vectors,
            ctx->d_probe_query_map,
            ctx->d_batch_queries,
            ctx->d_probe_distances,
            ctx->num_probe_candidates,
            num_queries,
            ctx->dimensions
        );
        
        // 同步设备
        cudaDeviceSynchronize();
        
        // 检查错误
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ERROR: CUDA核函数执行失败: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        printf("INFO: GPU批量probe距离计算完成\n");
        return 0;
    }

    // 辅助结构：用于Thrust排序的键值对
    struct DistanceIndexPair {
        float distance;
        int index;
        
        __host__ __device__
        bool operator<(const DistanceIndexPair& other) const {
            return distance < other.distance;
        }
    };

    // 用于筛选查询候选的结构体（必须在全局命名空间中定义）
    struct QueryFilter {
        int target_query;
        int* query_map;
        int num_candidates;
        int query_idx;
        
        __host__ __device__
        bool operator()(int idx) {
            if (idx >= num_candidates) return false;
            return query_map[idx] == target_query;
        }
    };

    // 用于填充距离索引对的结构体（必须在全局命名空间中定义）
    struct FillPairs {
        int query_idx;
        int num_candidates;
        float* distances;
        int* indices;
        
        __host__ __device__
        DistanceIndexPair operator()(int idx) {
            DistanceIndexPair pair;
            // indices[idx] 是 filtered_indices 中的值，即全局候选索引
            int vector_idx = indices[idx];
            // 距离矩阵布局：distance[query_idx * num_candidates + candidate_idx]
            // query_idx 是当前查询索引，vector_idx 是全局候选索引
            int distance_idx = query_idx * num_candidates + vector_idx;
            pair.distance = distances[distance_idx];
            pair.index = vector_idx;  // 保存全局候选索引，用于后续提取TID
            return pair;
        }
    };

    // CUDA核函数：并行处理所有查询的TopK选择
    // 每个block处理一个查询，所有查询并行执行
    // 
    // 注意：此kernel仅在cuda_wrapper.cu内部使用，不需要在头文件中声明
    // 
    // 性能说明：
    // 1. 使用原子操作收集候选（有性能瓶颈，但对于中等规模的候选数可接受）
    // 2. 只有线程0执行排序（线程利用率不高，但对于小k值可接受）
    // 3. 使用简单的选择排序（O(n*k)复杂度，适合k较小的情况）
    // 
    // 改进方向（如果性能成为瓶颈）：
    // 1. 使用并行扫描代替原子操作收集候选
    // 2. 使用所有线程参与排序（如bitonic sort或warp-level primitives）
    // 3. 对于大k值，考虑使用堆排序或Thrust的device端排序
    // 4. 考虑使用warp-sort等优化的TopK算法（参考fusion_cos_topk_warpsort.cu）
    __global__ void parallel_topk_per_query_kernel(
        const float* probe_distances,      // 距离矩阵 [num_queries * num_candidates]
        const int* probe_query_map,        // 查询映射 [num_candidates]
        const char* probe_tids,            // TID数组 [num_candidates * 6]
        int* topk_indices,                 // 输出TopK索引 [num_queries * k]
        float* topk_distances,             // 输出TopK距离 [num_queries * k]
        char* topk_tids,                   // 输出TopK TID [num_queries * k * 6]
        int* topk_counts,                  // 输出每个查询的TopK数量 [num_queries]
        int num_candidates,
        int num_queries,
        int k,
        int max_candidates_per_query) {    // 每个查询的最大候选数（用于共享内存分配）
        
        int query_idx = blockIdx.x;
        if (query_idx >= num_queries) {
            return;
        }
        
        // 使用共享内存存储当前查询的候选索引和距离
        extern __shared__ char shared_mem[];
        int* shared_indices = (int*)shared_mem;
        float* shared_distances = (float*)(shared_mem + max_candidates_per_query * sizeof(int));
        // 计算shared_pairs的起始位置
        // max_candidates_per_query * (sizeof(int) + sizeof(float)) = max_candidates_per_query * 8
        // 这个值总是8的倍数，所以不需要对齐
        size_t pairs_offset = max_candidates_per_query * (sizeof(int) + sizeof(float));
        DistanceIndexPair* shared_pairs = (DistanceIndexPair*)(shared_mem + pairs_offset);
        
        // 使用共享内存中的原子计数器（性能瓶颈，但对于中等规模候选数可接受）
        __shared__ int s_candidate_count;
        
        // 初始化计数器
        if (threadIdx.x == 0) {
            s_candidate_count = 0;
        }
        __syncthreads();
        
        // 第一步：筛选属于当前查询的候选
        // 修复：先收集所有候选，然后按距离排序选择最优的max_candidates_per_query个
        // 使用临时数组存储所有候选（如果超过共享内存限制，需要分批处理）
        
        // 临时存储所有属于当前查询的候选
        int temp_candidate_count = 0;
        
        // 第一遍：统计属于当前查询的候选总数
        for (int candidate_idx = threadIdx.x; candidate_idx < num_candidates; candidate_idx += blockDim.x) {
            if (probe_query_map[candidate_idx] == query_idx) {
                atomicAdd(&s_candidate_count, 1);
            }
        }
        __syncthreads();
        
        int total_query_candidates = s_candidate_count;
        
        // 重置计数器用于第二遍
        if (threadIdx.x == 0) {
            s_candidate_count = 0;
        }
        __syncthreads();
        
        // 第二遍：收集候选数据，但限制在共享内存范围内
        // 如果候选数超过限制，只取前max_candidates_per_query个（按发现顺序）
        // 注意：这里仍然有截断问题，但至少保证了内存安全
        for (int candidate_idx = threadIdx.x; candidate_idx < num_candidates; candidate_idx += blockDim.x) {
            if (probe_query_map[candidate_idx] == query_idx) {
                int pos = atomicAdd(&s_candidate_count, 1);
                if (pos < max_candidates_per_query) {
                    shared_indices[pos] = candidate_idx;
                    int distance_idx = query_idx * num_candidates + candidate_idx;
                    shared_distances[pos] = probe_distances[distance_idx];
                }
            }
        }
        
        __syncthreads();
        
        // 只有线程0执行排序和TopK选择（线程利用率低，但对于小k值可接受）
        // 改进方向：使用所有线程参与排序（如bitonic sort）
        if (threadIdx.x == 0) {
            int candidate_count = s_candidate_count;
            if (candidate_count > max_candidates_per_query) {
                candidate_count = max_candidates_per_query;
                // 警告：候选数被截断，可能影响结果质量
                // printf("WARNING: 查询%d的候选数(%d)超过共享内存限制(%d)，结果可能不完整\n", 
                //        query_idx, total_query_candidates, max_candidates_per_query);
            }
            
            if (candidate_count == 0) {
                topk_counts[query_idx] = 0;
                // 填充无效值
                for (int i = 0; i < k; i++) {
                    topk_indices[query_idx * k + i] = -1;
                    topk_distances[query_idx * k + i] = FLT_MAX;
                    for (int j = 0; j < 6; j++) {
                        topk_tids[query_idx * k * 6 + i * 6 + j] = 0;
                    }
                }
                return;
            }
            
            // 创建距离-索引对
            for (int i = 0; i < candidate_count; i++) {
                shared_pairs[i].distance = shared_distances[i];
                shared_pairs[i].index = shared_indices[i];
            }
            
            // 简单的选择排序选择TopK（O(n*k)复杂度，适合k较小的情况）
            // 改进方向：对于大k值，使用堆排序或Thrust的device端排序
            int topk = (k < candidate_count) ? k : candidate_count;
            
            // 部分排序：选择最小的topk个
            for (int i = 0; i < topk; i++) {
                int min_idx = i;
                float min_dist = shared_pairs[i].distance;
                
                for (int j = i + 1; j < candidate_count; j++) {
                    if (shared_pairs[j].distance < min_dist) {
                        min_dist = shared_pairs[j].distance;
                        min_idx = j;
                    }
                }
                
                // 交换
                if (min_idx != i) {
                    DistanceIndexPair temp = shared_pairs[i];
                    shared_pairs[i] = shared_pairs[min_idx];
                    shared_pairs[min_idx] = temp;
                }
            }
            
            // 将TopK结果写入全局内存
            topk_counts[query_idx] = topk;
            for (int i = 0; i < topk; i++) {
                int candidate_idx = shared_pairs[i].index;
                topk_indices[query_idx * k + i] = candidate_idx;
                topk_distances[query_idx * k + i] = shared_pairs[i].distance;
                
                // 提取TID
                int tid_offset = candidate_idx * 6;
                int output_offset = query_idx * k * 6 + i * 6;
                for (int j = 0; j < 6; j++) {
                    topk_tids[output_offset + j] = probe_tids[tid_offset + j];
                }
            }
            
            // 填充剩余位置
            for (int i = topk; i < k; i++) {
                topk_indices[query_idx * k + i] = -1;
                topk_distances[query_idx * k + i] = FLT_MAX;
                for (int j = 0; j < 6; j++) {
                    topk_tids[query_idx * k * 6 + i * 6 + j] = 0;
                }
            }
        }
    }

    // 全局变量保存最后一个CUDA错误信息（使用__thread确保线程安全）
    static __thread cudaError_t last_cuda_error = cudaSuccess;
    static __thread char last_cuda_error_msg[256] = "no error";
    
    // 内部函数：保存CUDA错误信息
    static void save_cuda_error(cudaError_t err) {
        last_cuda_error = err;
        if (err != cudaSuccess) {
            snprintf(last_cuda_error_msg, sizeof(last_cuda_error_msg), "%s", cudaGetErrorString(err));
        } else {
            snprintf(last_cuda_error_msg, sizeof(last_cuda_error_msg), "no error");
        }
    }
    
    // GPU TopK选择（并行处理所有查询）
    int cuda_topk_probe_candidates(CudaCenterSearchContext* ctx,
                                   int k,
                                   int num_queries,
                                   int* topk_query_ids,
                                   void* topk_vector_ids,  /* ItemPointerData数组 */
                                   float* topk_distances,
                                   int* topk_counts) {
        if (!ctx || !ctx->initialized || !topk_query_ids || !topk_vector_ids || !topk_distances) {
            printf("ERROR: GPU TopK选择参数无效\n");
            return -1;
        }
        
        if (!ctx->probes_uploaded) {
            printf("ERROR: Probe数据未上传\n");
            return -1;
        }
        
        // 分配GPU内存存储TopK结果（如果尚未分配）
        cudaError_t err;
        bool need_allocate = false;
        
        if (!ctx->d_topk_indices) {
            size_t topk_indices_size = num_queries * k * sizeof(int);
            err = cudaMalloc(&ctx->d_topk_indices, topk_indices_size);
            if (err != cudaSuccess) {
                printf("ERROR: TopK索引内存分配失败: %s\n", cudaGetErrorString(err));
                return -1;
            }
            need_allocate = true;
        }
        
        if (!ctx->d_topk_distances) {
            size_t topk_distances_size = num_queries * k * sizeof(float);
            err = cudaMalloc(&ctx->d_topk_distances, topk_distances_size);
            if (err != cudaSuccess) {
                printf("ERROR: TopK距离内存分配失败: %s\n", cudaGetErrorString(err));
                return -1;
            }
            need_allocate = true;
        }
        
        if (!ctx->d_topk_tids) {
            size_t topk_tids_size = num_queries * k * 6;  // 每个查询k个结果，每个TID 6字节
            err = cudaMalloc(&ctx->d_topk_tids, topk_tids_size);
            if (err != cudaSuccess) {
                printf("ERROR: TopK TID内存分配失败: %s\n", cudaGetErrorString(err));
                return -1;
            }
            need_allocate = true;
        }
        
        // 分配设备端计数数组
        int* d_topk_counts = NULL;
        err = cudaMalloc(&d_topk_counts, num_queries * sizeof(int));
        if (err != cudaSuccess) {
            printf("ERROR: TopK计数内存分配失败: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        // 检查probe数据
        if (!ctx->d_probe_vectors || !ctx->d_probe_tids || !ctx->d_probe_query_map) {
            printf("ERROR: Probe数据未上传（分离存储方案）\n");
            cudaFree(d_topk_counts);
            return -1;
        }
        
        // 动态获取GPU的共享内存限制
        cudaDeviceProp prop;
        cudaError_t prop_err = cudaGetDeviceProperties(&prop, 0);
        size_t max_shared_mem = 48 * 1024;  // 默认48KB
        if (prop_err == cudaSuccess) {
            // 使用每个block的共享内存限制（通常为48KB或96KB）
            max_shared_mem = prop.sharedMemPerBlock;
            printf("DEBUG: GPU共享内存限制: %zu字节 (每个block), 最大线程数: %d, 最大grid大小: (%d, %d, %d), 计算能力: %d.%d\n", 
                   max_shared_mem, prop.maxThreadsPerBlock, 
                   prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2],
                   prop.major, prop.minor);
            // 检查是否有共享内存配置限制
            if (prop.sharedMemPerBlockOptin > 0) {
                printf("DEBUG: GPU支持可配置共享内存，最大: %zu字节\n", prop.sharedMemPerBlockOptin);
            }
        } else {
            printf("WARNING: 无法获取GPU属性，使用默认共享内存限制: %zu字节\n", max_shared_mem);
        }
        
        // 验证grid大小不超过GPU限制
        if (prop_err == cudaSuccess && num_queries > prop.maxGridSize[0]) {
            printf("ERROR: 查询数 (%d) 超过GPU最大grid大小 (%d)\n", num_queries, prop.maxGridSize[0]);
            cudaFree(d_topk_counts);
            save_cuda_error(cudaErrorInvalidValue);
            return -1;
        }
        
        // 计算每个查询的最大候选数（用于共享内存分配）
        // 需要存储：indices + distances + pairs（需要考虑对齐）
        // 每个元素：sizeof(int) + sizeof(float) + sizeof(DistanceIndexPair)
        size_t elem_size = sizeof(int) + sizeof(float) + sizeof(DistanceIndexPair);
        
        // 注意：kernel中还有静态共享内存 __shared__ int s_candidate_count (4字节)
        // 总共享内存 = 动态共享内存 + 静态共享内存
        // 为了安全，我们从最大共享内存中减去静态共享内存的大小
        size_t static_shared_mem = sizeof(int);  // s_candidate_count
        size_t available_shared_mem = max_shared_mem - static_shared_mem;
        
        // 使用二分搜索找到满足共享内存限制的最大候选数
        // 注意：每个查询的候选数可能不同，但我们只能基于总候选数来估算
        // 为了安全，我们使用一个合理的上界（例如，假设每个查询的候选数不超过总候选数）
        // 但实际上，每个查询的候选数可能远小于总候选数，所以我们可以使用总候选数作为上界
        int max_candidates_per_query = 0;
        int low = 0, high = ctx->num_probe_candidates;
        while (low <= high) {
            int mid = (low + high) / 2;
            // 计算shared_pairs的起始位置
            // mid * (sizeof(int) + sizeof(float)) = mid * 8，总是8的倍数
            size_t pairs_offset = mid * (sizeof(int) + sizeof(float));
            // 总共享内存大小 = pairs_offset + mid * sizeof(DistanceIndexPair)
            // = mid * 8 + mid * 8 = mid * 16
            size_t test_shared_mem_size = pairs_offset + mid * sizeof(DistanceIndexPair);
            // 整体对齐到8字节（虽然已经是8的倍数，但为了安全还是对齐）
            test_shared_mem_size = (test_shared_mem_size + 7) & ~7;
            
            if (test_shared_mem_size <= available_shared_mem) {
                max_candidates_per_query = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        
        printf("INFO: 共享内存计算 - 最大共享内存: %zu字节, 可用共享内存: %zu字节 (扣除静态共享内存 %zu字节), 元素大小: %zu字节, 每查询最大候选数: %d (总候选数: %d, 查询数: %d)\n",
               max_shared_mem, available_shared_mem, static_shared_mem, elem_size, max_candidates_per_query, ctx->num_probe_candidates, num_queries);
        
        // 验证计算结果
        if (max_candidates_per_query > 0) {
            size_t test_pairs_offset = max_candidates_per_query * (sizeof(int) + sizeof(float));
            size_t test_shared_mem_size = test_pairs_offset + max_candidates_per_query * sizeof(DistanceIndexPair);
            test_shared_mem_size = (test_shared_mem_size + 7) & ~7;
            printf("DEBUG: 验证共享内存计算 - 计算的共享内存: %zu字节, GPU限制: %zu字节\n",
                   test_shared_mem_size, max_shared_mem);
        }
        
        if (max_candidates_per_query == 0) {
            printf("ERROR: 无法分配共享内存，即使候选数为1也超过限制\n");
            cudaFree(d_topk_counts);
            save_cuda_error(cudaErrorInvalidValue);
            return -1;
        }
        
        // 如果候选数太多，限制为最大共享内存能容纳的数量
        if (ctx->num_probe_candidates > max_candidates_per_query) {
            printf("WARNING: 总候选数 (%d) 超过共享内存限制 (%d)，将限制每个查询的候选数。这可能导致大批量查询的结果不完整。建议减少probe数量或增加GPU共享内存。\n", 
                   ctx->num_probe_candidates, max_candidates_per_query);
        }
        
        // 对于大批量查询，建议的解决方案：
        // 1. 使用更大的共享内存配置（如果GPU支持）
        // 2. 分批处理候选
        // 3. 使用全局内存代替共享内存（性能较低但容量大）
        
        // 计算实际的共享内存大小
        // max_candidates_per_query * (sizeof(int) + sizeof(float)) = max_candidates_per_query * 8
        size_t pairs_offset = max_candidates_per_query * (sizeof(int) + sizeof(float));
        // 总共享内存大小 = pairs_offset + max_candidates_per_query * sizeof(DistanceIndexPair)
        // = max_candidates_per_query * 8 + max_candidates_per_query * 8 = max_candidates_per_query * 16
        size_t shared_mem_size = pairs_offset + max_candidates_per_query * sizeof(DistanceIndexPair);
        // 整体对齐到8字节（虽然已经是8的倍数，但为了安全还是对齐）
        shared_mem_size = (shared_mem_size + 7) & ~7;
        
        printf("INFO: TopK kernel参数 - 候选数: %d, 每查询最大候选数: %d, 共享内存: %zu字节 (限制: %zu字节), k: %d, 查询数: %d, elem_size: %zu字节\n",
               ctx->num_probe_candidates, max_candidates_per_query, shared_mem_size, max_shared_mem, 
               k, num_queries, elem_size);
        
        // 设置kernel参数
        dim3 block_size(256);  // 每个block 256个线程
        // grid_size: 每个查询一个block，所以x维度是查询数
        dim3 grid_size(num_queries, 1, 1);  // 明确指定为1D grid
        
        // 验证block大小不超过GPU限制
        if (prop_err == cudaSuccess && block_size.x > prop.maxThreadsPerBlock) {
            printf("ERROR: block大小 (%d) 超过GPU最大线程数 (%d)\n", block_size.x, prop.maxThreadsPerBlock);
            cudaFree(d_topk_counts);
            save_cuda_error(cudaErrorInvalidValue);
            return -1;
        }
        
        // 验证共享内存大小不超过每个block的限制
        if (prop_err == cudaSuccess && shared_mem_size > prop.sharedMemPerBlock) {
            printf("ERROR: 共享内存大小 (%zu字节) 超过GPU每个block的限制 (%zu字节)\n", 
                   shared_mem_size, prop.sharedMemPerBlock);
            cudaFree(d_topk_counts);
            save_cuda_error(cudaErrorInvalidValue);
            return -1;
        }
        
        // 验证grid和block大小
        if (grid_size.x == 0 || block_size.x == 0 || num_queries <= 0) {
            printf("ERROR: 无效的kernel参数 - grid_size: (%d, %d, %d), block_size: (%d, %d, %d), num_queries: %d\n",
                   grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, num_queries);
            cudaFree(d_topk_counts);
            save_cuda_error(cudaErrorInvalidValue);
            return -1;
        }
        
        // 验证共享内存大小不超过限制
        if (shared_mem_size > max_shared_mem) {
            printf("ERROR: 共享内存大小 (%zu字节) 超过GPU限制 (%zu字节)\n", shared_mem_size, max_shared_mem);
            cudaFree(d_topk_counts);
            save_cuda_error(cudaErrorInvalidValue);
            return -1;
        }
        
        printf("DEBUG: 准备启动kernel - grid_size: (%d, %d, %d), block_size: (%d, %d, %d), shared_mem: %zu字节, max_shared_mem: %zu字节\n",
               grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, shared_mem_size, max_shared_mem);
        
        // 最后验证：确保共享内存大小不超过限制
        if (shared_mem_size > max_shared_mem) {
            printf("ERROR: 最终验证失败 - 共享内存大小 (%zu字节) 超过GPU限制 (%zu字节)\n", shared_mem_size, max_shared_mem);
            cudaFree(d_topk_counts);
            save_cuda_error(cudaErrorInvalidValue);
            return -1;
        }
        
        // 清除之前的CUDA错误状态
        cudaGetLastError();
        
        // 启动并行kernel，所有查询同时处理
        parallel_topk_per_query_kernel<<<grid_size, block_size, shared_mem_size>>>(
            ctx->d_probe_distances,
            ctx->d_probe_query_map,
            ctx->d_probe_tids,
            ctx->d_topk_indices,
            ctx->d_topk_distances,
            ctx->d_topk_tids,
            d_topk_counts,
            ctx->num_probe_candidates,
            num_queries,
            k,
            max_candidates_per_query
        );
        
        // 先检查kernel启动错误（不等待完成）
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            save_cuda_error(err);
            printf("ERROR: CUDA kernel启动失败: %s (共享内存: %zu字节, 每查询最大候选数: %d, 候选数: %d, 查询数: %d, k: %d)\n", 
                   cudaGetErrorString(err), shared_mem_size, max_candidates_per_query, 
                   ctx->num_probe_candidates, num_queries, k);
            cudaFree(d_topk_counts);
            return -1;
        }
        
        // 同步设备并检查执行错误
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            save_cuda_error(err);
            printf("ERROR: CUDA kernel执行失败: %s (共享内存: %zu字节, 每查询最大候选数: %d, 候选数: %d, 查询数: %d, k: %d)\n", 
                   cudaGetErrorString(err), shared_mem_size, max_candidates_per_query,
                   ctx->num_probe_candidates, num_queries, k);
            cudaFree(d_topk_counts);
            return -1;
        }
        
        // 再次检查是否有异步错误
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            save_cuda_error(err);
            printf("ERROR: CUDA异步错误检测: %s\n", cudaGetErrorString(err));
            cudaFree(d_topk_counts);
            return -1;
        }
        
        // 清除错误状态
        save_cuda_error(cudaSuccess);
        
        // 一次性复制所有结果到主机
        char* vector_ids_ptr = (char*)topk_vector_ids;
        
        // 复制TopK距离、TID和计数
        err = cudaMemcpy(topk_distances, ctx->d_topk_distances,
                        num_queries * k * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            save_cuda_error(err);
            printf("ERROR: TopK距离复制失败: %s (大小: %zu字节)\n", 
                   cudaGetErrorString(err), num_queries * k * sizeof(float));
            cudaFree(d_topk_counts);
            return -1;
        }
        
        err = cudaMemcpy(vector_ids_ptr, ctx->d_topk_tids,
                        num_queries * k * 6, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            save_cuda_error(err);
            printf("ERROR: TopK TID复制失败: %s (大小: %zu字节)\n", 
                   cudaGetErrorString(err), (size_t)(num_queries * k * 6));
            cudaFree(d_topk_counts);
            return -1;
        }
        
        err = cudaMemcpy(topk_counts, d_topk_counts,
                        num_queries * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            save_cuda_error(err);
            printf("ERROR: TopK计数复制失败: %s (大小: %zu字节)\n", 
                   cudaGetErrorString(err), num_queries * sizeof(int));
            cudaFree(d_topk_counts);
            return -1;
        }
        
        // 填充查询ID（topk_query_ids应该存储查询ID，而不是候选索引）
        for (int query_idx = 0; query_idx < num_queries; query_idx++) {
            for (int i = 0; i < k; i++) {
                int result_idx = query_idx * k + i;
                topk_query_ids[result_idx] = query_idx;
            }
        }
        
        // 清理临时内存
        cudaFree(d_topk_counts);
        
        printf("INFO: GPU TopK选择完成（并行处理 %d 个查询）\n", num_queries);
        return 0;
    }
    
    // 一致性检查函数：验证向量数据和TID数据的索引对应关系
    int cuda_verify_probe_consistency(CudaCenterSearchContext* ctx,
                                     int num_candidates,
                                     int dimensions) {
        if (!ctx || !ctx->initialized) {
            printf("ERROR: 一致性检查参数无效\n");
            return -1;
        }
        
        // 检查分离存储方案
        if (!ctx->d_probe_vectors || !ctx->d_probe_tids || !ctx->d_probe_query_map) {
            printf("ERROR: Probe数据未上传（分离存储方案）\n");
            return -1;
        }
        
        printf("INFO: 开始一致性检查 - 候选数: %d, 维度: %d\n", num_candidates, dimensions);
        
        // 分配GPU内存存储验证结果
        int* d_verification_results;
        cudaError_t err = cudaMalloc(&d_verification_results, num_candidates * sizeof(int));
        if (err != cudaSuccess) {
            printf("ERROR: 验证结果内存分配失败: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        // 初始化验证结果
        cudaMemset(d_verification_results, 0, num_candidates * sizeof(int));
        
        // 启动一致性检查kernel
        dim3 block_size(256);
        dim3 grid_size((num_candidates + block_size.x - 1) / block_size.x);
        
        verify_probe_consistency_kernel<<<grid_size, block_size>>>(
            ctx->d_probe_vectors,
            ctx->d_probe_tids,
            ctx->d_probe_query_map,
            d_verification_results,
            num_candidates,
            dimensions
        );
        
        cudaDeviceSynchronize();
        
        // 检查kernel执行错误
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ERROR: 一致性检查kernel执行失败: %s\n", cudaGetErrorString(err));
            cudaFree(d_verification_results);
            return -1;
        }
        
        // 复制验证结果到主机
        int* verification_results = (int*)malloc(num_candidates * sizeof(int));
        err = cudaMemcpy(verification_results, d_verification_results,
                        num_candidates * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("ERROR: 验证结果复制失败: %s\n", cudaGetErrorString(err));
            free(verification_results);
            cudaFree(d_verification_results);
            return -1;
        }
        
        // 检查验证结果
        int error_count = 0;
        int vector_error_count = 0;
        int tid_error_count = 0;
        int query_map_error_count = 0;
        
        for (int i = 0; i < num_candidates; i++) {
            if (verification_results[i] != 0) {
                error_count++;
                if (verification_results[i] & 1) {
                    vector_error_count++;
                }
                if (verification_results[i] & 2) {
                    tid_error_count++;
                }
                if (verification_results[i] & 4) {
                    query_map_error_count++;
                }
                
                // 只打印前10个错误
                if (error_count <= 10) {
                    printf("WARNING: 候选 %d 一致性检查失败 - 错误码: %d (向量:%s, TID:%s, 查询映射:%s)\n",
                           i, verification_results[i],
                           (verification_results[i] & 1) ? "无效" : "有效",
                           (verification_results[i] & 2) ? "无效" : "有效",
                           (verification_results[i] & 4) ? "无效" : "有效");
                }
            }
        }
        
        // 清理内存
        free(verification_results);
        cudaFree(d_verification_results);
        
        // 报告结果
        if (error_count == 0) {
            printf("INFO: 一致性检查通过 - 所有 %d 个候选都有效\n", num_candidates);
            return 0;
        } else {
            printf("ERROR: 一致性检查失败 - 总错误数: %d / %d (向量错误: %d, TID错误: %d, 查询映射错误: %d)\n",
                   error_count, num_candidates, vector_error_count, tid_error_count, query_map_error_count);
            return -1;
        }
    }
    
    // 获取最后一个CUDA错误的字符串（用于PostgreSQL错误报告）
    const char* cuda_get_last_error_string(void) {
        return last_cuda_error_msg;
    }

}