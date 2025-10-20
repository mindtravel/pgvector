#include <limits>
#include <type_traits>
#include <math_constants.h>

#include "l2norm.cuh"
#include "pch.h"
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
    
    /* 按照 laneId 访问数据 */
    const T* row_input = input + row * len;
    for (int i = warp_id * kWarpSize + lane; i < len; i += n_warps * kWarpSize) {
        queue.add(row_input[i], static_cast<IdxT>(i));
    }
    
    /* 把 buffer 中剩余数合并到 queue 中 */
    queue.done();
    
    /* 将 queue 中的数存储到显存中（所有线程都要调用）*/
    if (warp_id == 0) {
        T* row_out_val = output_vals + row * k;
        IdxT* row_out_idx = output_idx + row * k;
        queue.store(row_out_val, row_out_idx);
    }
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
    if (k > kMaxCapacity) {
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
        if (capacity <= 32) {
            select_k_kernel<64, true, T, IdxT><<<grid, block, 0, stream>>>(
                input, batch_size, len, k, output_vals, output_idx);
        } else if (capacity <= 64) {
            select_k_kernel<128, true, T, IdxT><<<grid, block, 0, stream>>>(
                input, batch_size, len, k, output_vals, output_idx);
        } else {
            select_k_kernel<256, true, T, IdxT><<<grid, block, 0, stream>>>(
                input, batch_size, len, k, output_vals, output_idx);
        }
    } else {
        if (capacity <= 32) {
            select_k_kernel<64, false, T, IdxT><<<grid, block, 0, stream>>>(
                input, batch_size, len, k, output_vals, output_idx);
        } else if (capacity <= 64) {
            select_k_kernel<128, false, T, IdxT><<<grid, block, 0, stream>>>(
                input, batch_size, len, k, output_vals, output_idx);
        } else {
            select_k_kernel<256, false, T, IdxT><<<grid, block, 0, stream>>>(
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

} // namespace warpsort
} // namespace pgvector
