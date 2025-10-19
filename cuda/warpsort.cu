/**
 * Warp-Sort Top-K Implementation for pgvector
 * 
 * Based on RAFT (RAPIDS AI) warp-sort implementation:
 * - raft/cpp/include/raft/matrix/detail/select_warpsort.cuh
 * - raft/cpp/include/raft/util/bitonic_sort.cuh
 * 
 * This implementation provides GPU-accelerated top-k selection using
 * warp-level primitives and bitonic sorting networks.
 * 
 * Key features:
 * - Support for k up to 256 (kMaxCapacity)
 * - Warp-level parallelism using shuffle operations
 * - Register-based storage for minimal memory overhead
 * - Bitonic merge network for efficient sorting
 * 
 * Copyright (c) 2024, pgvector
 * Adapted from RAFT (Apache 2.0 License)
 */

#include <limits>
#include <type_traits>
#include <math_constants.h>

#include "kernels.h"
#include "pch.h"
#include "warpsort_utils.cuh"
#include "bitonic.cuh"


namespace pgvector {
namespace warpsort {

using namespace warpsort_utils;
using namespace bitonic;

// ============================================================================
// Warp Sort Base Class
// ============================================================================

/**
 * 基于warp级的top-k选择优先队列基类。
 * 
 * 使用寄存器存储，在一个warp内维护k个最小（或最大）值。
 * 每个线程保存 Capacity/kWarpWidth 个元素。
 * 
 * 注意：这个基类主要用于存储数组和提供 store 方法。
 * 实际的 top-k 选择逻辑在 WarpSortFiltered 中实现。
 * 
 * @tparam Capacity 队列最大容量（必须为2的幂，且不超过kMaxCapacity）
 * @tparam Ascending 为true时选择最小k个，为false时选择最大k个
 * @tparam T 值类型（通常为float）
 * @tparam IdxT 索引类型（通常为int或uint32_t）
 */
template<int Capacity, bool Ascending, typename T, typename IdxT>
class WarpSort {
    static_assert(isPowerOf2(Capacity), "Capacity must be power of 2");
    static_assert(Capacity <= kMaxCapacity, "Capacity exceeds maximum");
    
public:
    static constexpr int kWarpWidth = (Capacity < kWarpSize) ? Capacity : kWarpSize;
    static constexpr int kMaxArrLen = Capacity / kWarpWidth;
    
    /* 哨兵值（dummy value）用于初始化和填充 */
    static __device__ __forceinline__ T kDummy()
    {
        return Ascending ? upper_bound<T>() : lower_bound<T>();
    }
    
    const int k;  /* 需要选择的元素个数 */
    
    __device__ WarpSort(int k_val) : k(k_val)
    {
        #pragma unroll
        for (int i = 0; i < kMaxArrLen; i++) {
            val_arr_[i] = kDummy();
            idx_arr_[i] = IdxT{};
        }
    }
    
    /**
     * 将 queue 中结果存储到显存
     * 
     * 注意：只存储前 k 个元素（queue 部分）
     */
    __device__ void store(T* out_val, IdxT* out_idx) const
    {
        int idx = Pow2<kWarpWidth>::mod(laneId());
        
        #pragma unroll
        for (int i = 0; i < kMaxArrLen && idx < k; i++, idx += kWarpWidth) {
            out_val[idx] = val_arr_[i];
            out_idx[idx] = idx_arr_[i];
        }
    }
    
protected:
    T val_arr_[kMaxArrLen];
    IdxT idx_arr_[kMaxArrLen];
};

// ============================================================================
// Warp Sort Filtered (Optimized for Large Inputs)
// ============================================================================

/**
 * WarpSortFiltered：针对大输入序列优化的warp-sort。
 * 
 * - val_arr_ 的前半部分作为 queue（存储当前 top-k）
 * - val_arr_ 的后半部分作为 buffer（临时存储新元素）
 * - buffer 满时，对整个 val_arr_ 进行排序，确保不丢失任何数据
 * 
 * 算法流程：
 * 1. 过滤：仅当元素优于当前第k值时才添加到 buffer
 * 2. buffer 满时，对整个数组（queue + buffer）排序
 * 3. 排序后，前半部分是最好的元素，后半部分清空重用
 * 4. 动态更新第k值的阈值
 * 
 * @tparam Capacity 总容量（必须 >= 64，这样 kMaxArrLen >= 2）
 */
template<int Capacity, bool Ascending, typename T, typename IdxT>
class WarpSortFiltered : public WarpSort<Capacity, Ascending, T, IdxT> {
    using Base = WarpSort<Capacity, Ascending, T, IdxT>;
    
public:
    using Base::kWarpWidth;
    using Base::kMaxArrLen;
    using Base::k;
    
    /* 
     * 数组布局（动态计算）：
     * - val_arr_[0 .. k_arr_len_-1]: queue 区域（存储当前 top-k）
     * - val_arr_[k_arr_len_ .. kMaxArrLen-1]: buffer 区域（临时存储新元素）
     * 
     * 关键设计：
     * - queue 大小由 k 决定：k_arr_len_ = ceil(k / kWarpWidth)
     * - buffer 大小 = kMaxArrLen - k_arr_len_
     * - 确保 queue 足够大来存储所有 k 个元素
     */
    static_assert(kMaxArrLen >= 2, "Capacity must be >= 64 for WarpSortFiltered");
    
    __device__ WarpSortFiltered(int k_val, T limit = Base::kDummy())
        : Base(k_val), buf_len_(0), k_th_(limit)
    {
        /* 计算需要多少个数组元素来存储 k 个结果 */
        k_arr_len_ = (k + kWarpWidth - 1) / kWarpWidth;
        
        /* 确保不超过总容量 */
        // if (k_arr_len_ >= kMaxArrLen) {
        //     k_arr_len_ = kMaxArrLen - 1;  /* 至少留1个给buffer */
        // }
        
        /* val_arr_ 已在基类构造函数中初始化为 kDummy */
    }
    
    /**
     * 添加一个键值对。
     * 先过滤，通过过滤的元素添加到 buffer 区域。
     * 
     * 重要：merge 操作是跨所有线程的，必须所有线程同步参与，
     * 即使某些线程当前没有要添加的元素。
     */
    __device__ void add(T val, IdxT idx)
    {
        /* 计算 buffer 的最大长度 */
        int buf_max_len = kMaxArrLen - k_arr_len_;
        
        /* 
         * 关键修复：将 merge 检查移到 do_add 之外
         * 确保所有线程都参与 merge 判断和执行
         */
        if (any(buf_len_ >= buf_max_len)) {
            merge_buf_();
        }
        
        /* 判断是否要 filter 这个数 */
        bool do_add = is_ordered<Ascending>(val, k_th_);
           
        if (do_add) {
            /* 添加键值对到 buffer 区域 */
            add_to_buf_(val, idx);
        }
    }
    
    /**
     * 完成所有添加操作，处理 buffer 中的剩余元素
     */
    __device__ void done()
    {
        if (any(buf_len_ != 0)) {
            merge_buf_();
        }
    }
    
private:
    using Base::val_arr_;
    using Base::idx_arr_;
    
    int buf_len_;    /* 当前 buffer 中的元素个数 */
    int k_arr_len_;  /* queue 区域的长度（存储 k 个元素需要的数组长度）*/
    T k_th_;         /* 当前第 k 个元素的值（过滤阈值）*/
    
    /**
     * 将元素添加到 buffer 区域（val_arr_ 的后半部分）
     * 
     * 注意：buffer 从 val_arr_[k_arr_len_] 开始
     */
    __device__ __forceinline__ void add_to_buf_(T val, IdxT idx)
    {
        /* 
         * 使用编译时常量索引避免寄存器溢出
         * 编译器会将循环展开为多个 if 语句，每个使用常量索引
         * 
         * 注意：由于 k_arr_len_ 是运行时变量，我们需要遍历整个数组
         */
        #pragma unroll
        for (int i = 0; i < kMaxArrLen; i++) {
            /* buffer 从 k_arr_len_ 开始 */
            if (i == k_arr_len_ + buf_len_) {
                val_arr_[i] = val;
                idx_arr_[i] = idx;
            }
        }
        buf_len_++;
    }
    
    /**
     * 合并 buffer 到 queue：对整个数组排序
     * 
     * 核心思想：
     * 1. 对整个 val_arr_[0..kMaxArrLen-1] 进行 bitonic sort
     * 2. 排序后，前面的元素是最好的（最小或最大）
     * 3. 清空 buffer 区域（从 k_arr_len_ 开始），等待下一批数据
     * 4. 更新过滤阈值 k_th_
     */
    __device__ __forceinline__ void merge_buf_()
    {
        /* 对整个数组排序（queue + buffer 一起排序）*/
        Bitonic<kMaxArrLen>(Ascending, kWarpWidth).sort(val_arr_, idx_arr_);
        
        /* 清空 buffer 区域（从 k_arr_len_ 开始），为下一批数据做准备 */
        #pragma unroll
        for (int i = 0; i < kMaxArrLen; i++) {
            if (i >= k_arr_len_) {
                val_arr_[i] = Base::kDummy();
                idx_arr_[i] = IdxT{};
            }
        }
        
        /* 重置 buffer 计数 */
        buf_len_ = 0;
        
        /* 更新第 k 个元素的阈值 */
        set_k_th_();
    }
    
    /**
     * 更新过滤阈值：第 k 个元素的值
     * 
     * 注意：排序后，第 k 个元素分布在 warp 的某个线程中
     * 需要通过 shuffle 广播到所有线程
     */
    __device__ __forceinline__ void set_k_th_()
    {
        /* 
         * 计算第 k 个元素在数组中的位置
         * - 假设 k=16, kWarpWidth=32
         * - 第16个元素在：thread_id = (16-1) % 32 = 15, arr_idx = (16-1) / 32 = 0
         * - 即：thread 15 的 val_arr_[0]
         */
        int k_thread = (k - 1) % kWarpWidth;  /* 第 k 个元素所在的线程 */
        int k_idx = (k - 1) / kWarpWidth;      /* 第 k 个元素在该线程数组中的索引 */
        
        /* 将第 k 个元素从所在线程广播到所有线程 */
        k_th_ = shfl(val_arr_[k_idx], k_thread, kWarpWidth);
    }
};

// ============================================================================
// Public API: Top-K Selection Kernel
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
