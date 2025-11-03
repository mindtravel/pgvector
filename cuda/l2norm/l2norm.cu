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
        if (lane == 0) {
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
 * 优化版本3: 使用 float4 向量化加载（适用于维度是4的倍数的情况）
 * 
 * 优势：
 * - 每次加载 4 个 float，提高内存带宽利用率
 * - 减少循环次数
 */
__global__ void l2_norm_kernel_optimized_v3(
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
    
    const float4* vec_ptr4 = reinterpret_cast<const float4*>(vectors + bid * n_dim);
    const float* vec_ptr = vectors + bid * n_dim;
    
    float sum = 0.0f;
    const int vec4_count = n_dim / 4;
    const int remainder = n_dim % 4;
    
    /* 向量化加载 float4 */
    #pragma unroll 4
    for (int i = 0; i < vec4_count; i += blockDim.x) {
        int idx = i + tid;
        if (idx < vec4_count) {
            /* 使用 __ldg() 优化只读访问 */
            float4 val4 = __ldg(&vec_ptr4[idx]);
            sum += val4.x * val4.x + val4.y * val4.y + 
                   val4.z * val4.z + val4.w * val4.w;
        }
    }
    
    /* 处理余数（如果不是4的倍数） */
    if (remainder > 0) {
        int idx = vec4_count * 4 + tid;
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
    int idx = bid * n_dim + tid;

    // 计算当前向量的平方和
    float square = 0.0f;
    if (tid < n_dim) {
        square = vectors[idx] * vectors[idx];
        // printf("%d: %f %f\n", bid, vectors[idx], square);
    }

    shared_mem[tid] = square;
    __syncthreads();

    // 规约求和（修复：处理n_dim不是2的幂的情况）
    // 使用安全的规约方法，确保所有元素都被处理
    for (int s = 1; s < n_dim; s *= 2) {
        __syncthreads();
        if (tid % (2 * s) == 0 && (tid + s) < n_dim) {
            shared_mem[tid] += shared_mem[tid + s];
        }
    }
    
    // 计算L2范数
    if (tid == 0) {
        vector_l2_norm[bid] = sqrt(shared_mem[0]);
        // printf("%d: %f\n", bid, vector_l2_norm[bid]);
    }
}
    


