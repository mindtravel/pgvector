#include <limits>
#include <type_traits>
#include <math_constants.h>

#include "../l2norm/l2norm.cuh"
#include "../pch.h"
#include "warpsort_utils.cuh"
#include "warpsort.cuh"
#include "bitonic.cuh"

namespace pgvector {
namespace warpsort_topk {

using namespace warpsort_utils;
using namespace warpsort;

// ============================================================================
// Public API: Warpsort Top-K Selection Kernel
// ============================================================================

/**
 * 从矩阵每一行中选取 top-k 最小或最大元素。
 * 
 * 每个 CUDA block 处理一行。每个 block 内的 warp 独立地使用 WarpSortFiltered 算法完成排序筛选。
 * 
 * @param[in] input        输入矩阵，形状为 [batch_size, len]
 * @param[in] batch_size   行数（批大小）
 * @param[in] len          每行的元素个数
 * @param[in] k            选取的元素个数
 * @param[out] output_vals 输出 top-k 值，形状为 [batch_size, k]
 * @param[out] output_idx  输出 top-k 对应的索引，形状为 [batch_size, k]
 * @param[in] select_min   若为 true 选取最小的 k 个，否则选取最大的 k 个
 */
template<int Capacity, bool Ascending, typename T, typename IdxT>
__global__ void select_k_kernel(
    const T* __restrict__ input,
    int batch_size,
    int len,
    int k,
    T* __restrict__ output_vals,
    IdxT* __restrict__ output_idx)
{
    const int row = blockIdx.x;
    if (row >= batch_size) return;
    
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = laneId();
    const int n_warps = blockDim.x / kWarpSize; /* 一共参与WarpSort的数量（目前都是1） */
    
    /* 在一个warp中，建立一个长度为k的队列 */
    WarpSortFiltered<Capacity, Ascending, T, IdxT> queue(k); 
    
    /* 确保所有线程都执行到这里 */
    __syncwarp();
    
    /* 
     * 关键修复：使用固定次数的循环，确保所有线程执行相同次数的迭代
     * 这对于 WarpSortFiltered 的 any() 和 __any_sync() 同步是必需的
     * 
     * 问题：原来的条件循环 `for (int i = ...; i < len; i += ...)` 导致不同线程
     * 执行不同次数的迭代，使得 queue.add() 的调用不同步，导致 __any_sync() 死锁
     * 
     * 解决方案：计算最大迭代次数，所有线程执行相同次数的循环，
     * 每个迭代都同步调用 queue.add()（无论是否有有效数据）
     */
    
    /* 计算最大迭代次数：ceil(len / (n_warps * kWarpSize)) */
    int max_iter = (len + n_warps * kWarpSize - 1) / (n_warps * kWarpSize);
    
    /* 按照 laneId 访问数据 */
    const T* row_input = input + row * len;
    
    for (int iter = 0; iter < max_iter; iter++) {
        /* 在每个迭代开始时同步 */
        __syncwarp();
        
        /* 计算当前迭代处理的索引 */
        int i = warp_id * kWarpSize + lane + iter * n_warps * kWarpSize;
        
        /* 如果索引有效，则添加到队列；否则添加 dummy 值 */
        if (i < len) {
        queue.add(row_input[i], static_cast<IdxT>(i));
        } else {
            /* 使用 WarpSort 基类的静态方法获取正确的dummy值 */
            using BaseWarpSort = WarpSort<Capacity, Ascending, T, IdxT>;
            const T dummy_val = BaseWarpSort::kDummy();
            queue.add(dummy_val, static_cast<IdxT>(-1));
        }
    }
    
    /* 把 buffer 中剩余数合并到 queue 中 */
    queue.done();
    
    /* 同步 warp，确保 done() 完成后再 store */
    __syncwarp();
    
    /* 将 queue 中的数存储到显存中（所有线程都要调用）*/
        T* row_out_val = output_vals + row * k;
        IdxT* row_out_idx = output_idx + row * k;
        queue.store(row_out_val, row_out_idx);
}

/**
 * Host function to launch top-k selection.
 * Automatically chooses appropriate capacity based on k.
 */
template<typename T, typename IdxT>
cudaError_t select_k(
    const T* input,
    int batch_size,
    int len,
    int k,
    T* output_vals,
    IdxT* output_idx,
    bool select_min,
    cudaStream_t stream = 0)
{
    // 参数验证
    if (k <= 0 || k > kMaxCapacity) {
        return cudaErrorInvalidValue;
    }
    if (batch_size <= 0) {
        return cudaErrorInvalidValue;
    }
    if (len <= 0) {
        return cudaErrorInvalidValue;
    }
    if (input == nullptr || output_vals == nullptr || output_idx == nullptr) {
        return cudaErrorInvalidValue;
    }
    
    // CUDA grid 维度限制检查
    if (batch_size > 2147483647) {
        return cudaErrorInvalidValue;
    }
    
    /* 
     * 选择合适的 Capacity
     * 
     * WarpSortFiltered 需要 buffer 空间，因此：
     * - Capacity 必须 > k（不能等于 k）
     * - 最小使用 64（确保 kMaxArrLen >= 2）
     * - 选择最小的满足 Capacity > k 的 2 的幂
     */
    int capacity = 32;  /* 最小使用 32 */
    while (capacity < k) capacity <<= 1;  /* 注意：必须 > k，不能等于 */
    
    dim3 block(32);  /* 使用32线程（单个warp）*/
    dim3 grid(batch_size);
    
    /* 模板的非类型参数必须是常量，所以只能用这一系列分支来使用不同尺寸的函数 */
    if (select_min) {
        if (capacity < 64) {
            select_k_kernel<64, true, T, IdxT><<<grid, block, 0, stream>>>(
                input, batch_size, len, k, output_vals, output_idx);
        } else if (capacity < 128) {
            select_k_kernel<128, true, T, IdxT><<<grid, block, 0, stream>>>(
                input, batch_size, len, k, output_vals, output_idx);
        } else if (capacity < 256){
            select_k_kernel<256, true, T, IdxT><<<grid, block, 0, stream>>>(
                input, batch_size, len, k, output_vals, output_idx);
        } else {
            select_k_kernel<512, true, T, IdxT><<<grid, block, 0, stream>>>(
                input, batch_size, len, k, output_vals, output_idx);
        }
    } else {
        if (capacity < 64) {
            select_k_kernel<64, false, T, IdxT><<<grid, block, 0, stream>>>(
                input, batch_size, len, k, output_vals, output_idx);
        } else if (capacity < 128) {
            select_k_kernel<128, false, T, IdxT><<<grid, block, 0, stream>>>(
                input, batch_size, len, k, output_vals, output_idx);
        } else if (capacity < 256){
            select_k_kernel<256, false, T, IdxT><<<grid, block, 0, stream>>>(
                input, batch_size, len, k, output_vals, output_idx);
        } else {
            select_k_kernel<512, false, T, IdxT><<<grid, block, 0, stream>>>(
                input, batch_size, len, k, output_vals, output_idx);
        }
    }
    
    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t select_k<float, int>(
    const float*, int, int, int, float*, int*, bool, cudaStream_t);

template cudaError_t select_k<float, uint32_t>(
    const float*, int, int, int, float*, uint32_t*, bool, cudaStream_t);

} // namespace warpsort_topk
} // namespace pgvector
