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
    
}