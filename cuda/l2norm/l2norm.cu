/**
 * l2norm_optimized.cu
 * 优化版本的 L2 范数计算 kernel
 * 
 * 改进点：
 * 1. 向量化加载 (vectorized load) 提高内存带宽
 * 2. Warp-level 规约优化小维度情况
 * 3. 寄存器级规约减少共享内存访问
 * 4. 使用 __ldg() 优化只读访问
 * 5. 更高效的规约模式
 */

#include "l2norm.cuh"
#include <device_launch_parameters.h>
#include <cmath>
#include "pch.h"

/* 
 * 优化版本: 使用向量化加载 + warp-level 规约
 * 
 * 优势：
 * - 使用 __ldg() 优化只读访问
 * - 对于 n_dim <= 32，直接在 warp 内规约，无需共享内存
 * - 对于 32 < n_dim <= 128，使用 warp-level 规约 + 共享内存合并
 * - 对于 n_dim > 128，使用优化的共享内存规约
 */
__global__ void l2_norm_kernel(
    const float* __restrict__ vectors, 
    float* __restrict__ vector_l2_norm, 
    int n_batch, 
    int n_dim) 
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid & 31;  // lane within warp
    const int warp_id = tid / 32;
    const int n_warps = (blockDim.x + 31) / 32;
    
    if (bid >= n_batch) return;
    
    const float* vec_ptr = vectors + bid * n_dim;
    
    /* 方案1: 如果维度 <= 32，使用 warp-level 规约（最优） */
    if (n_dim <= 32) {
        /* 只有第一个 warp 处理数据 */
        if (warp_id == 0) {
            float sum = 0.0f;
            
            /* 每个 lane 处理对应的元素 */
            if (lane < n_dim) {
                /* 使用 __ldg() 优化只读访问 */
                float val = __ldg(&vec_ptr[lane]);
                sum = val * val;
            }
            
            /* Warp-level 规约使用 shuffle 操作 */
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            
            /* Lane 0 写入结果 */
            if (lane == 0) {
                vector_l2_norm[bid] = sqrtf(sum);
            }
        }
        return;
    }
    
    /* 方案2: 32 < n_dim <= 128，使用 warp-level 规约 + 共享内存合并 */
    if (n_dim <= 128) {
        extern __shared__ float sdata[];
        float sum = 0.0f;
        
        /* 每个线程处理多个元素（向量化加载） */
        const int elems_per_thread = (n_dim + blockDim.x - 1) / blockDim.x;
        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            int idx = tid + i * blockDim.x;
            if (idx < n_dim) {
                float val = __ldg(&vec_ptr[idx]);
                sum += val * val;
            }
        }
        
        /* Warp-level 规约 */
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        /* 将每个 warp 的结果写入共享内存 */
        if (lane == 0) {
            sdata[warp_id] = sum;
        }
        __syncthreads();
        
        /* Warp 0 合并所有 warp 的结果 */
        if (warp_id == 0) {
            sum = (lane < n_warps) ? sdata[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane == 0) {
                vector_l2_norm[bid] = sqrtf(sum);
            }
        }
        return;
    }
    
    /* 方案3: n_dim > 128，使用 warp-level 规约 + 共享内存合并（与方案2相同策略） */
    extern __shared__ float sdata[];
    
    /* 计算平方和（每个线程处理多个元素） */
    float sum = 0.0f;
    const int elems_per_thread = (n_dim + blockDim.x - 1) / blockDim.x;
    
    #pragma unroll 4
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < n_dim) {
            /* 使用 __ldg() 优化只读访问 */
            float val = __ldg(&vec_ptr[idx]);
            sum += val * val;
        }
    }
    
    /* Warp-level 规约 */
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    /* 将每个 warp 的结果写入共享内存 */
    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    
    /* Warp 0 合并所有 warp 的结果 */
    if (warp_id == 0) {
        sum = (lane < n_warps) ? sdata[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            vector_l2_norm[bid] = sqrtf(sum);
        }
    }
}

/* 
 * 优化版本2: 简化的高效版本（推荐用于大多数情况）
 * 
 * 策略：
 * - 小维度 (<=32): warp shuffle 规约
 * - 中等维度 (33-256): warp shuffle + shared memory 合并
 * - 大维度 (>256): 优化的共享内存规约
 */
__global__ void l2_norm_kernel_optimized_v2(
    const float* __restrict__ vectors, 
    float* __restrict__ vector_l2_norm, 
    int n_batch, 
    int n_dim) 
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid / 32;
    const int n_warps = (blockDim.x + 31) / 32;
    
    if (bid >= n_batch) return;
    
    const float* vec_ptr = vectors + bid * n_dim;
    
    /* 计算平方和（向量化加载） */
    float sum = 0.0f;
    const int elems_per_thread = (n_dim + blockDim.x - 1) / blockDim.x;
    
    #pragma unroll 4
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < n_dim) {
            /* 使用 __ldg() 优化只读访问 */
            float val = __ldg(&vec_ptr[idx]);
            sum += val * val;
        }
    }
    
    /* Warp-level 规约 */
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    /* 如果只有一个 warp 或维度很小，直接写入结果 */
    if (n_warps == 1 || n_dim <= 32) {
        /* 只有第一个 warp 写入结果 */
        if (warp_id == 0 && lane == 0) {
            vector_l2_norm[bid] = sqrtf(sum);
        }
        return;
    }
    
    /* 多个 warp 需要合并：使用共享内存 */
    extern __shared__ float sdata[];
    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    
    /* Warp 0 负责最终合并 */
    if (warp_id == 0) {
        sum = (lane < n_warps) ? sdata[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            vector_l2_norm[bid] = sqrtf(sum);
        }
    }
}

/* 
 * Device函数：使用 float4 向量化加载计算单个向量的L2范数（FMA优化版本）
 * 
 * 优势：
 * - 每次加载 4 个 float，提高内存带宽利用率
 * - 使用FMA指令减少指令数和延迟
 * - 可作为device函数被其他kernel调用
 * 
 * @param vec_ptr 向量数据的起始指针
 * @param n_dim 向量维度
 * @param tid 线程ID
 * @param block_dim block大小
 * @return 该线程计算的平方和（局部结果）
 */
__device__ __forceinline__ float compute_l2_norm_float4_device(
    const float* __restrict__ vec_ptr,
    int n_dim,
    int tid,
    int block_dim)
{
    const float4* vec_ptr4 = reinterpret_cast<const float4*>(vec_ptr);
    const int lane = tid & 31;
    const int warp_id = tid / 32;
    const int n_warps = (block_dim + 31) / 32;
    
    float sum = 0.0f;
    const int vec4_count = n_dim / 4;
    const int remainder = n_dim % 4;
    
    /* 向量化加载 float4，使用FMA指令优化 */
    #pragma unroll 4
    for (int i = 0; i < vec4_count; i += block_dim) {
        int idx = i + tid;
        if (idx < vec4_count) {
            /* 使用 __ldg() 优化只读访问 */
            float4 val4 = __ldg(&vec_ptr4[idx]);
            /* 使用FMA指令：融合乘加操作，减少指令数 */
            sum += val4.x * val4.x + val4.y * val4.y + 
       val4.z * val4.z + val4.w * val4.w;
        }
    }
    
    /* 处理余数（如果不是4的倍数），使用FMA */
    if (remainder > 0) {
        int idx = vec4_count * 4 + tid;
        if (idx < n_dim) {
            float val = __ldg(&vec_ptr[idx]);
            sum = fmaf(val, val, sum);
        }
    }
    
    /* Warp-level 规约 */
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    /* 合并多个 warp 的结果 */
    if (n_warps > 1) {
        extern __shared__ float sdata[];
        if (lane == 0) {
            sdata[warp_id] = sum;
        }
        __syncthreads();
        
        if (warp_id == 0) {
            sum = (lane < n_warps) ? sdata[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
        }
    }
    
    return sum;
}

/* 
 * Kernel：使用 float4 向量化加载（FMA优化版本）
 * 调用device函数进行计算
 */
__global__ void l2_norm_kernel_float4(
    const float* __restrict__ vectors, 
    float* __restrict__ vector_l2_norm, 
    int n_batch, 
    int n_dim) 
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid / 32;
    
    if (bid >= n_batch) return;
    
    const float* vec_ptr = vectors + bid * n_dim;
    
    /* 调用device函数计算平方和 */
    float sum = compute_l2_norm_float4_device(vec_ptr, n_dim, tid, blockDim.x);
    
    /* 写入结果 */
    if (lane == 0 && warp_id == 0) {
        vector_l2_norm[bid] = sqrtf(sum);
    }
}

/*
* 计算模长
* Args:
*   vectors: 原始向量组
*   vetcor_suqared_sum: 规范化后向量组的l2 norm
*   n_dim: 向量维数 
*   n_batch: 一组中向量个数 
*  
* 在向量的分量上并行计算元素的平方，再使用树形规约，最后sqrt
* 速度更快但是无法保留原始向量信息
*/
__global__ void l2_norm_kernel_basic(
    float *vectors, 
    float *vector_l2_norm, 
    int n_batch, 
    int n_dim
) 
{
    extern __shared__ float shared_mem[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_dim = blockDim.x;
    
    // 计算当前向量的平方和
    float square = 0.0f;
    if (tid < n_dim) {
        int idx = bid * n_dim + tid;
        square = vectors[idx] * vectors[idx];
    }

    // 初始化共享内存（只初始化需要的部分）
    if (tid < block_dim) {
        shared_mem[tid] = (tid < n_dim) ? square : 0.0f;
    }
    __syncthreads();

    // 规约求和（处理block_size可能大于n_dim的情况）
    // 规约范围应该是实际的n_dim，而不是block_dim
    int reduce_size = block_dim;
    if (reduce_size > n_dim) {
        reduce_size = n_dim;
    }
    
    for (int s = 1; s < reduce_size; s *= 2) {
        __syncthreads();
        if (tid % (2 * s) == 0 && (tid + s) < reduce_size) {
            shared_mem[tid] += shared_mem[tid + s];
        }
    }
    
    // 计算L2范数
    if (tid == 0) {
        vector_l2_norm[bid] = sqrtf(shared_mem[0]);
    }
}

/**
 * 统一的L2范数计算host函数实现
 * 
 * 提供自动选择和手动切换两种模式
 */
void compute_l2_norm_gpu(
    const float* vectors,
    float* vector_l2_norm,
    int n_batch,
    int n_dim,
    L2NormVersion version,
    cudaStream_t stream)
{
    /* 计算kernel配置参数 */
    const int block_size = 256;  /* 推荐block大小 */
    const int grid_size = n_batch;
    
    /* 计算共享内存大小（某些kernel需要） */
    const int shared_mem_size = (n_dim > 32) ? 
        ((block_size + 31) / 32) * sizeof(float) : 0;
    
    /* 手动指定版本 */
    if (version != L2NORM_AUTO) {
        switch (version) {
            case L2NORM_BASIC: {
                /* basic kernel需要block大小至少等于n_dim */
                /* 使用min(1024, max(n_dim, 256))来平衡性能和限制 */
                const int basic_block_size = (n_dim <= 256) ? 256 : ((n_dim <= 1024) ? n_dim : 256);
                const int basic_shared_mem = basic_block_size * sizeof(float);
                if (stream) {
                    l2_norm_kernel_basic<<<grid_size, basic_block_size, 
                        basic_shared_mem, stream>>>(
                        const_cast<float*>(vectors), vector_l2_norm, n_batch, n_dim);
                } else {
                    l2_norm_kernel_basic<<<grid_size, basic_block_size, 
                        basic_shared_mem>>>(
                        const_cast<float*>(vectors), vector_l2_norm, n_batch, n_dim);
                }
                break;
            }
                
            case L2NORM_OPTIMIZED:
                if (stream) {
                    l2_norm_kernel<<<grid_size, block_size, 
                        shared_mem_size, stream>>>(
                        vectors, vector_l2_norm, n_batch, n_dim);
                } else {
                    l2_norm_kernel<<<grid_size, block_size, 
                        shared_mem_size>>>(
                        vectors, vector_l2_norm, n_batch, n_dim);
                }
                break;
                
            case L2NORM_OPTIMIZED_V2:
                if (stream) {
                    l2_norm_kernel_optimized_v2<<<grid_size, block_size, 
                        shared_mem_size, stream>>>(
                        vectors, vector_l2_norm, n_batch, n_dim);
                } else {
                    l2_norm_kernel_optimized_v2<<<grid_size, block_size, 
                        shared_mem_size>>>(
                        vectors, vector_l2_norm, n_batch, n_dim);
                }
                break;
                
            case L2NORM_OPTIMIZED_V3:
                if (stream) {
                    l2_norm_kernel_float4<<<grid_size, block_size, 
                        shared_mem_size, stream>>>(
                        vectors, vector_l2_norm, n_batch, n_dim);
                } else {
                    l2_norm_kernel_float4<<<grid_size, block_size, 
                        shared_mem_size>>>(
                        vectors, vector_l2_norm, n_batch, n_dim);
                }
                break;
                
            default:
                /* 不应该到达这里，但为了安全起见使用默认策略 */
                version = L2NORM_AUTO;
                break;
        }
        
        if (version != L2NORM_AUTO) {
            return;
        }
    }
    
    /* 自动选择最优版本 */
    /* 策略1: 如果dim是4的倍数且较大，优先使用float4向量化版本 */
    if (n_dim >= 128 && (n_dim % 4 == 0)) {
        if (stream) {
            l2_norm_kernel_float4<<<grid_size, block_size, 
                shared_mem_size, stream>>>(
                vectors, vector_l2_norm, n_batch, n_dim);
        } else {
            l2_norm_kernel_float4<<<grid_size, block_size, 
                shared_mem_size>>>(
                vectors, vector_l2_norm, n_batch, n_dim);
        }
        return;
    }
    
    /* 策略2: 对于其他情况，使用优化版本2（简化高效版本） */
    /* 该版本内部会根据dim自动选择最优策略 */
    if (stream) {
        l2_norm_kernel_optimized_v2<<<grid_size, block_size, 
            shared_mem_size, stream>>>(
            vectors, vector_l2_norm, n_batch, n_dim);
    } else {
        l2_norm_kernel_optimized_v2<<<grid_size, block_size, 
            shared_mem_size>>>(
            vectors, vector_l2_norm, n_batch, n_dim);
    }
}


