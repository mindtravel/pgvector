#include <cuda_runtime.h>
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
#include <cstddef>  // for offsetof
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
            
            // 清理probe候选数据
            if (ctx->d_probe_index_tuples) {
                cudaFree(ctx->d_probe_index_tuples);
                ctx->d_probe_index_tuples = NULL;
                printf("INFO: IndexTuple数据内存释放成功\n");
            }
            
            if (ctx->d_probe_tuple_offsets) {
                cudaFree(ctx->d_probe_tuple_offsets);
                ctx->d_probe_tuple_offsets = NULL;
                printf("INFO: IndexTuple偏移量内存释放成功\n");
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
            
            // 清理IndexTuple存储
            if (ctx->d_probe_index_tuples) {
                cudaFree(ctx->d_probe_index_tuples);
                ctx->d_probe_index_tuples = NULL;
                printf("INFO: IndexTuple数据内存释放成功\n");
            }
            
            if (ctx->d_probe_tuple_offsets) {
                cudaFree(ctx->d_probe_tuple_offsets);
                ctx->d_probe_tuple_offsets = NULL;
                printf("INFO: IndexTuple偏移量内存释放成功\n");
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

    // CUDA核函数：从IndexTuple中提取向量并计算L2距离（与CPU存储方式完全一致）
    // IndexTuple结构：t_tid(6字节) + t_info(2字节) + Vector数据
    // Vector结构：vl_len_(4字节) + dim(2字节) + unused(2字节) + x[dimensions]
    __global__ void compute_batch_probe_l2_distances_from_index_tuples_kernel(
                                                            const char* probe_index_tuples,
                                                            const int* probe_tuple_offsets,
                                                            const int* probe_query_map,
                                                            const float* batch_queries,
                                                            float* batch_distances,
                                                            int num_candidates,
                                                            int num_queries,
                                                            int dimensions,
                                                            size_t fixed_tuple_size) {
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
        
        // 计算IndexTuple的偏移量
        int tuple_offset;
        if (fixed_tuple_size > 0) {
            // 固定大小：直接计算偏移
            tuple_offset = candidate_idx * fixed_tuple_size;
        } else {
            // 变长：使用偏移量数组
            tuple_offset = probe_tuple_offsets[candidate_idx];
        }
        
        // IndexTuple布局：t_tid(6字节) + t_info(2字节) + Vector数据
        // 注意：IndexTupleData 结构体中 t_tid 是第一个字段，t_info 是第二个字段
        // 跳过前8字节（t_tid + t_info），直接访问Vector数据
        const char* tuple_ptr = probe_index_tuples + tuple_offset;
        const char* vector_ptr = tuple_ptr + 8;  // 跳过t_tid(6) + t_info(2)
        
        // Vector布局：vl_len_(4) + dim(2) + unused(2) + x[dimensions]
        // 跳过Vector头部8字节，直接访问x[]数组
        const float* vector_data = (const float*)(vector_ptr + 8);
        
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

    // 上传完整的IndexTuple数据到GPU（与CPU端存储方式完全一致）
    // IndexTuple包含：t_info + t_tid + Vector数据
    int cuda_upload_probe_vectors(CudaCenterSearchContext* ctx,
                                  const char* index_tuples,
                                  const int* query_ids,
                                  const size_t* tuple_sizes,
                                  const int* tuple_offsets,
                                  int num_candidates,
                                  int dimensions,
                                  size_t fixed_tuple_size) {
        if (!ctx || !ctx->initialized || !index_tuples || !query_ids) {
            printf("ERROR: 上传IndexTuple数据参数无效\n");
            return -1;
        }
        
        printf("INFO: 开始上传 %d 个IndexTuple到GPU (与CPU存储方式一致), 维度: %d\n", 
               num_candidates, dimensions);
        
        // 清理旧数据
        if (ctx->d_probe_index_tuples) {
            cudaFree(ctx->d_probe_index_tuples);
            ctx->d_probe_index_tuples = NULL;
        }
        if (ctx->d_probe_tuple_offsets) {
            cudaFree(ctx->d_probe_tuple_offsets);
            ctx->d_probe_tuple_offsets = NULL;
        }
        if (ctx->d_probe_query_map) {
            cudaFree(ctx->d_probe_query_map);
            ctx->d_probe_query_map = NULL;
        }
        
        cudaError_t err;
        
        // 计算总内存大小
        size_t total_size = 0;
        int* offsets = NULL;
        
        if (fixed_tuple_size > 0) {
            // 固定大小：所有元组大小相同
            total_size = num_candidates * fixed_tuple_size;
            ctx->index_tuple_size = fixed_tuple_size;
            printf("INFO: 使用固定元组大小: %zu字节\n", fixed_tuple_size);
        } else if (tuple_sizes != NULL) {
            // 变长元组：每个元组大小不同
            offsets = (int*)malloc((num_candidates + 1) * sizeof(int));
            if (!offsets) {
                printf("ERROR: 无法分配偏移量数组\n");
                return -1;
            }
            
            offsets[0] = 0;
            for (int i = 0; i < num_candidates; i++) {
                offsets[i + 1] = offsets[i] + tuple_sizes[i];
            }
            total_size = offsets[num_candidates];
            ctx->index_tuple_size = 0;  // 0表示变长
            printf("INFO: 使用变长元组，总大小: %zu字节\n", total_size);
        } else if (tuple_offsets != NULL) {
            // 使用提供的偏移量
            offsets = (int*)malloc((num_candidates + 1) * sizeof(int));
            if (!offsets) {
                printf("ERROR: 无法分配偏移量数组\n");
                return -1;
            }
            memcpy(offsets, tuple_offsets, num_candidates * sizeof(int));
            // 计算最后一个偏移量后的总大小
            if (tuple_sizes != NULL) {
                total_size = offsets[num_candidates - 1] + tuple_sizes[num_candidates - 1];
            } else {
                printf("ERROR: 提供偏移量时必须提供元组大小\n");
                free(offsets);
                return -1;
            }
            ctx->index_tuple_size = 0;
        } else {
            printf("ERROR: 必须提供fixed_tuple_size或tuple_sizes\n");
            return -1;
        }
        
        // 分配GPU内存存储IndexTuple数据
        err = cudaMalloc(&ctx->d_probe_index_tuples, total_size);
        if (err != cudaSuccess) {
            printf("ERROR: IndexTuple数据GPU内存分配失败: %s (需要%zu字节)\n", 
                   cudaGetErrorString(err), total_size);
            if (offsets) free(offsets);
            return -1;
        }
        
        // 分配GPU内存存储偏移量（用于变长元组）
        if (ctx->index_tuple_size == 0) {
            err = cudaMalloc(&ctx->d_probe_tuple_offsets, (num_candidates + 1) * sizeof(int));
            if (err != cudaSuccess) {
                printf("ERROR: IndexTuple偏移量GPU内存分配失败: %s\n", cudaGetErrorString(err));
                cudaFree(ctx->d_probe_index_tuples);
                ctx->d_probe_index_tuples = NULL;
                if (offsets) free(offsets);
                return -1;
            }
            
            // 上传偏移量
            err = cudaMemcpy(ctx->d_probe_tuple_offsets, offsets, 
                           (num_candidates + 1) * sizeof(int), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                printf("ERROR: IndexTuple偏移量上传失败: %s\n", cudaGetErrorString(err));
                cudaFree(ctx->d_probe_index_tuples);
                cudaFree(ctx->d_probe_tuple_offsets);
                ctx->d_probe_index_tuples = NULL;
                ctx->d_probe_tuple_offsets = NULL;
                if (offsets) free(offsets);
                return -1;
            }
        }
        
        // 上传IndexTuple数据
        if (fixed_tuple_size > 0) {
            // 固定大小：直接上传
            err = cudaMemcpy(ctx->d_probe_index_tuples, index_tuples, 
                           total_size, cudaMemcpyHostToDevice);
        } else {
            // 变长：需要打包到连续内存
            char* packed_tuples = (char*)malloc(total_size);
            if (!packed_tuples) {
                printf("ERROR: 无法分配打包缓冲区\n");
                cudaFree(ctx->d_probe_index_tuples);
                if (ctx->d_probe_tuple_offsets) {
                    cudaFree(ctx->d_probe_tuple_offsets);
                    ctx->d_probe_tuple_offsets = NULL;
                }
                if (offsets) free(offsets);
                return -1;
            }
            
            // 打包元组到连续内存
            for (int i = 0; i < num_candidates; i++) {
                size_t size = tuple_sizes[i];
                // 如果提供了tuple_offsets，使用它；否则使用我们计算的offsets
                int src_offset = tuple_offsets ? tuple_offsets[i] : offsets[i];
                const char* src = index_tuples + src_offset;
                memcpy(packed_tuples + offsets[i], src, size);
            }
            
            // 上传打包后的数据
            err = cudaMemcpy(ctx->d_probe_index_tuples, packed_tuples, 
                           total_size, cudaMemcpyHostToDevice);
            free(packed_tuples);
        }
        
        if (err != cudaSuccess) {
            printf("ERROR: IndexTuple数据上传失败: %s\n", cudaGetErrorString(err));
            cudaFree(ctx->d_probe_index_tuples);
            ctx->d_probe_index_tuples = NULL;
            if (ctx->d_probe_tuple_offsets) {
                cudaFree(ctx->d_probe_tuple_offsets);
                ctx->d_probe_tuple_offsets = NULL;
            }
            if (offsets) free(offsets);
            return -1;
        }
        
        // 上传查询映射
        size_t ids_size = num_candidates * sizeof(int);
        err = cudaMalloc(&ctx->d_probe_query_map, ids_size);
        if (err != cudaSuccess) {
            printf("ERROR: Probe查询映射内存分配失败: %s\n", cudaGetErrorString(err));
            cudaFree(ctx->d_probe_index_tuples);
            ctx->d_probe_index_tuples = NULL;
            if (ctx->d_probe_tuple_offsets) {
                cudaFree(ctx->d_probe_tuple_offsets);
                ctx->d_probe_tuple_offsets = NULL;
            }
            if (offsets) free(offsets);
            return -1;
        }
        
        err = cudaMemcpy(ctx->d_probe_query_map, query_ids, ids_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("ERROR: Probe查询映射上传失败: %s\n", cudaGetErrorString(err));
            cudaFree(ctx->d_probe_index_tuples);
            cudaFree(ctx->d_probe_query_map);
            ctx->d_probe_index_tuples = NULL;
            ctx->d_probe_query_map = NULL;
            if (ctx->d_probe_tuple_offsets) {
                cudaFree(ctx->d_probe_tuple_offsets);
                ctx->d_probe_tuple_offsets = NULL;
            }
            if (offsets) free(offsets);
            return -1;
        }
        
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
            printf("ERROR: Probe距离结果内存分配失败: %s\n", cudaGetErrorString(err));
            cudaFree(ctx->d_probe_index_tuples);
            ctx->d_probe_index_tuples = NULL;
            if (ctx->d_probe_query_map) {
                cudaFree(ctx->d_probe_query_map);
                ctx->d_probe_query_map = NULL;
            }
            if (ctx->d_probe_tuple_offsets) {
                cudaFree(ctx->d_probe_tuple_offsets);
                ctx->d_probe_tuple_offsets = NULL;
            }
            if (offsets) free(offsets);
            return -1;
        }
        
        if (offsets) free(offsets);
        
        printf("INFO: 成功上传 %d 个IndexTuple到GPU (总大小: %zu字节)\n", 
               num_candidates, total_size);
        printf("INFO: IndexTuple包含: t_info + t_tid + Vector数据 (与CPU存储方式一致)\n");
        
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
        
        // 使用IndexTuple核函数计算距离
        compute_batch_probe_l2_distances_from_index_tuples_kernel<<<grid_size, block_size>>>(
            ctx->d_probe_index_tuples,
            ctx->d_probe_tuple_offsets,
            ctx->d_probe_query_map,
            ctx->d_batch_queries,
            ctx->d_probe_distances,
            ctx->num_probe_candidates,
            num_queries,
            ctx->dimensions,
            ctx->index_tuple_size
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

    // GPU TopK选择（使用Thrust库进行GPU端排序）
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
        
        // 为每个查询进行TopK选择
        for (int query_idx = 0; query_idx < num_queries; query_idx++) {
            // 步骤1：在GPU上筛选属于该查询的候选（使用Thrust::copy_if）
            // 创建索引序列
            thrust::device_vector<int> indices(ctx->num_probe_candidates);
            thrust::sequence(indices.begin(), indices.end());
            
            // 筛选属于该查询的索引
            thrust::device_vector<int> filtered_indices(ctx->num_probe_candidates);
            
            // 创建筛选谓词：检查query_map是否等于query_idx
            QueryFilter predicate;
            predicate.target_query = query_idx;
            predicate.query_map = ctx->d_probe_query_map;
            predicate.num_candidates = ctx->num_probe_candidates;
            predicate.query_idx = query_idx;
            
            auto new_end = thrust::copy_if(indices.begin(), indices.end(), 
                                          filtered_indices.begin(), predicate);
            int candidate_count = new_end - filtered_indices.begin();
            
            if (candidate_count == 0) {
                topk_counts[query_idx] = 0;
                continue;
            }
            
            // 调整filtered_indices大小
            filtered_indices.resize(candidate_count);
            
            // 步骤2：提取距离并创建键值对
            thrust::device_vector<DistanceIndexPair> pairs(candidate_count);
            
            // 使用transform填充键值对
            FillPairs filler;
            filler.query_idx = query_idx;
            filler.num_candidates = ctx->num_probe_candidates;
            filler.distances = ctx->d_probe_distances;
            filler.indices = thrust::raw_pointer_cast(filtered_indices.data());
            
            thrust::transform(thrust::counting_iterator<int>(0),
                            thrust::counting_iterator<int>(candidate_count),
                            pairs.begin(), filler);
            
            // 步骤3：在GPU上排序
            thrust::sort(pairs.begin(), pairs.end());
            
            // 步骤4：取TopK并复制回主机
            int topk = (k < candidate_count) ? k : candidate_count;
            
            // 复制TopK结果到主机
            DistanceIndexPair* host_pairs = (DistanceIndexPair*)malloc(topk * sizeof(DistanceIndexPair));
            cudaMemcpy(host_pairs, thrust::raw_pointer_cast(pairs.data()),
                      topk * sizeof(DistanceIndexPair), cudaMemcpyDeviceToHost);
            
            // 步骤5：填充结果
            // 从IndexTuple中提取TID（t_tid在IndexTuple偏移量0字节处，ItemPointer是6字节）
            // 注意：IndexTupleData 结构体中 t_tid 是第一个字段
            // ItemPointer结构：BlockIdData(4字节) + OffsetNumber(2字节)
            // 直接复制完整的ItemPointer，保持真实TID
            char* vector_ids_ptr = (char*)topk_vector_ids;
            for (int i = 0; i < topk; i++) {
                int result_idx = query_idx * k + i;
                int candidate_idx = host_pairs[i].index;
                topk_query_ids[result_idx] = query_idx;
                topk_distances[result_idx] = host_pairs[i].distance;
                
                // 从GPU上的IndexTuple中提取TID
                // 计算IndexTuple的偏移量
                int tuple_offset;
                if (ctx->index_tuple_size > 0) {
                    tuple_offset = candidate_idx * ctx->index_tuple_size;
                } else {
                    // 从偏移量数组中获取
                    int offset;
                    cudaMemcpy(&offset, &ctx->d_probe_tuple_offsets[candidate_idx], 
                              sizeof(int), cudaMemcpyDeviceToHost);
                    tuple_offset = offset;
                }
                
                // 调试：打印关键信息（前几个结果）
                // 注意：printf 输出不会直接进入 PostgreSQL 日志，需要查看 stderr
                if (result_idx < 3) {
                    fprintf(stderr, "DEBUG: cuda_topk_probe_candidates - 结果 %d: query_idx=%d, candidate_idx=%d, tuple_offset=%d, k=%d\n",
                           result_idx, query_idx, candidate_idx, tuple_offset, k);
                }
                
                // IndexTuple布局：t_tid(6字节) + t_info(2字节) + Vector数据
                // 注意：IndexTupleData 结构体中 t_tid 是第一个字段，t_info 是第二个字段
                // 所以 t_tid 在偏移量 0 处，不是 2 处！
                // 直接复制完整的ItemPointer（6字节）到结果数组
                // ItemPointerData 是 6 字节：BlockIdData(4字节) + OffsetNumber(2字节)
                char temp_tid[6];
                cudaMemcpy(temp_tid,  // 临时缓冲区
                          ctx->d_probe_index_tuples + tuple_offset + 0,  // t_tid 在偏移量 0 处
                          6, cudaMemcpyDeviceToHost);
                
                // 调试：打印提取的 TID（前几个结果）
                if (result_idx < 3) {
                    // 从字节数组中提取 BlockNumber 和 OffsetNumber
                    // BlockIdData: bi_hi (uint16) + bi_lo (uint16)
                    // 注意：PostgreSQL 使用小端序
                    uint16_t bi_hi = *(uint16_t*)(temp_tid + 0);
                    uint16_t bi_lo = *(uint16_t*)(temp_tid + 2);
                    uint16_t offset = *(uint16_t*)(temp_tid + 4);
                    // BlockIdGetBlockNumber: ((uint32)bi_hi << 16) | bi_lo
                    uint32_t block = ((uint32_t)bi_hi << 16) | bi_lo;
                    fprintf(stderr, "DEBUG: cuda_topk_probe_candidates - 结果 %d: TID原始字节: [%02x %02x %02x %02x %02x %02x], bi_hi=%u, bi_lo=%u, block=%u, offset=%u\n",
                           result_idx,
                           (unsigned char)temp_tid[0], (unsigned char)temp_tid[1],
                           (unsigned char)temp_tid[2], (unsigned char)temp_tid[3],
                           (unsigned char)temp_tid[4], (unsigned char)temp_tid[5],
                           bi_hi, bi_lo, block, offset);
                }
                
                // 复制到结果数组
                memcpy(vector_ids_ptr + result_idx * 6, temp_tid, 6);
            }
            
            topk_counts[query_idx] = topk;
            free(host_pairs);
        }
        
        printf("INFO: GPU TopK选择完成\n");
        return 0;
    }

}