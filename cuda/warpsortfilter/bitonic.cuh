/**
 * Bitonic Sort Implementation for pgvector
 * 
 * Based on RAFT (RAPIDS AI) bitonic sort implementation:
 * - raft/cpp/include/raft/util/bitonic_sort.cuh
 * 
 * This implementation provides warp-level bitonic sorting operations
 * for GPU-accelerated top-k selection algorithms.
 * 
 * Copyright (c) 2024, pgvector
 * Adapted from RAFT (Apache 2.0 License)
 */

#ifndef BITONIC_CUH
#define BITONIC_CUH

#include "warpsort_utils.cuh"

namespace pgvector {
namespace bitonic {

/* 引入 warpsort_utils 命名空间中的工具函数和常量 */
using namespace warpsort_utils;

// ============================================================================
// Bitonic Sort Implementation
// ============================================================================

/**
 * Bitonic merge and sort operations for warp-level data.
 * 
 * Data is distributed across threads in a warp, with each thread holding
 * 'Size' elements. The total sorted data size is 'Size * warp_width'.
 * 
 * @tparam Size Number of elements per thread (must be power of 2)
 */
template<int Size = 1>
class Bitonic {
    static_assert(isPowerOf2(Size), "Size must be power of 2");
    
public:
    __device__ __forceinline__ Bitonic(bool ascending, int warp_width = kWarpSize)
        : ascending_(ascending), warp_width_(warp_width)
    {
    }
    
    /**
     * Bitonic Sort 核心操作：
     * 将两个按反方向排序的数组合并成一个有序的数组
     */
    template<typename KeyT, typename IdxT>
    __device__ __forceinline__ void merge(KeyT* keys, IdxT* indices) const
    {
        merge_impl(ascending_, warp_width_, keys, indices);
    }
    
    /**
     * 完整的排序操作
     */
    template<typename KeyT, typename IdxT>
    __device__ __forceinline__ void sort(KeyT* keys, IdxT* indices) const
    {
        sort_impl(ascending_, warp_width_, keys, indices);
    }
    
private:
    const bool ascending_;
    const int warp_width_;
    
    /* 允许不同 Size 的 Bitonic 实例访问彼此的私有成员 */
    template<int AnotherSize>
    friend class Bitonic;
    
    template<typename KeyT, typename IdxT>
    static __device__ __forceinline__ void merge_impl(
        bool ascending, int warp_width, KeyT* keys, IdxT* indices)
    {
        const int lane = laneId();
        
        /* 在线程内进行排序 */
        #pragma unroll
        for (int size = Size; size > 1; size >>= 1) {
            const int stride = size >> 1;
            
            #pragma unroll
            for (int offset = 0; offset < Size; offset += size) {
                #pragma unroll
                for (int i = 0; i < stride; i++) {
                    compare_and_swap(
                        ascending, 
                        keys[offset + i], 
                        keys[offset + i + stride],
                        indices[offset + i],
                        indices[offset + i + stride]
                    );
                }
            }
        }
        
        // Cross-thread comparisons using shuffle
        #pragma unroll
        for (int stride = warp_width >> 1; stride > 0; stride >>= 1) {
            #pragma unroll
            for (int offset = 0; offset < Size; offset++) {
                KeyT other_key = shfl(keys[offset], lane ^ stride, warp_width);
                IdxT other_idx = shfl(indices[offset], lane ^ stride, warp_width);
                
                /* 
                 * is_second: 当前线程是否在后半部分
                 * 比较逻辑与 RAFT 一致：
                 * - (ascending XOR is_second) 为 true 时：检查 key > other
                 * - (ascending XOR is_second) 为 false 时：检查 key < other
                 */
                const bool is_second = bool(lane & stride);
                const bool do_swap = (ascending != is_second) 
                    ? (keys[offset] > other_key) 
                    : (keys[offset] < other_key);
                
                if (do_swap) {
                    keys[offset] = other_key;
                    indices[offset] = other_idx;
                }
            }
        }
    }
    
    template<typename KeyT, typename IdxT>
    static __device__ __forceinline__ void sort_impl(
        bool ascending, int warp_width, KeyT* keys, IdxT* indices)
    {
        /**
         * Bitonic Sort 算法（基于 RAFT 实现）
         * 
         * 采用分治递归的思想：
         * 1. 如果 Size == 1，在 warp 级别逐步构建 bitonic 序列
         * 2. 如果 Size > 1：
         *    - 前半部分按降序排序
         *    - 后半部分按升序排序
         *    - 然后合并成一个完整的 bitonic 序列并排序
         * 
         * 由于 CUDA 不支持递归展开，我们需要手动展开递归
         */
        
        if constexpr (Size == 1) {
            /* Size == 1: 直接在 warp 级别构建 bitonic 序列 */
            const int lane = laneId();
            /* 逐步构建越来越大的 bitonic 序列 */
            for (int width = 2; width < warp_width; width <<= 1) {
                /* 
                 * 根据 lane 的位置决定排序方向
                 * lane & width 为 0 表示在前半部分，为 width 表示在后半部分
                 * 这样可以构建一个 bitonic 序列
                 */
                bool dir = bool(lane & width);
                merge_impl(dir, width, keys, indices);
            }
        } else if constexpr (Size == 2) {
            /* 
             * Size == 2: 手动展开递归逻辑
             * 
             * RAFT 递归逻辑：
             *   bitonic<1>::sort_impl(false, warp_width, keys);      // keys[0] 降序
             *   bitonic<1>::sort_impl(true, warp_width, keys+1);     // keys[1] 升序
             *   bitonic<2>::merge_impl(ascending, warp_width, keys); // 最后merge
             * 
             * 展开 bitonic<1>::sort_impl：
             *   for (width...) merge_impl(lane & width, width, keys);
             *   merge_impl(ascending, warp_width, keys);  // ← 这是关键！
             */
            const int lane = laneId();
            
            /* bitonic<1>::sort_impl(false, ..., keys[0]) */
            for (int width = 2; width < warp_width; width <<= 1) {
                bool dir = bool(lane & width);
                Bitonic<1>::merge_impl(dir, width, &keys[0], &indices[0]);
            }
            Bitonic<1>::merge_impl(false, warp_width, &keys[0], &indices[0]);  // 降序
            
            /* bitonic<1>::sort_impl(true, ..., keys[1]) */
            for (int width = 2; width < warp_width; width <<= 1) {
                bool dir = bool(lane & width);
                Bitonic<1>::merge_impl(dir, width, &keys[1], &indices[1]);
            }
            Bitonic<1>::merge_impl(true, warp_width, &keys[1], &indices[1]);  // 升序
        } else if constexpr (Size == 4) {
            /* 
             * Size == 4: 手动展开递归逻辑
             * 
             * RAFT 递归逻辑：
             *   bitonic<2>::sort_impl(false, warp_width, keys);      // keys[0:2] 降序
             *   bitonic<2>::sort_impl(true, warp_width, keys+2);     // keys[2:4] 升序
             *   bitonic<4>::merge_impl(ascending, warp_width, keys); // 最后merge
             * 
             * 手动展开 bitonic<2>::sort_impl：
             * - bitonic<1>::sort_impl(..., keys) + bitonic<1>::sort_impl(..., keys+1) + merge
             */
            const int lane = laneId();
            
            /* === 第一步：对 keys[0:2] 按降序排序（展开 bitonic<2>::sort_impl(false, ...)）=== */
            /* bitonic<1>::sort_impl(false, ..., keys[0]) */
            for (int width = 2; width < warp_width; width <<= 1) {
                bool dir = bool(lane & width);
                Bitonic<1>::merge_impl(dir, width, &keys[0], &indices[0]);
            }
            Bitonic<1>::merge_impl(false, warp_width, &keys[0], &indices[0]);
            
            /* bitonic<1>::sort_impl(true, ..., keys[1]) */
            for (int width = 2; width < warp_width; width <<= 1) {
                bool dir = bool(lane & width);
                Bitonic<1>::merge_impl(dir, width, &keys[1], &indices[1]);
            }
            Bitonic<1>::merge_impl(true, warp_width, &keys[1], &indices[1]);
            
            /* bitonic<2>::merge_impl(false) - 完成 bitonic<2>::sort_impl 的最后一步 */
            Bitonic<2>::merge_impl(false, warp_width, &keys[0], &indices[0]);
            
            /* === 第二步：对 keys[2:4] 按升序排序（展开 bitonic<2>::sort_impl(true, ...)）=== */
            /* bitonic<1>::sort_impl(false, ..., keys[2]) */
            for (int width = 2; width < warp_width; width <<= 1) {
                bool dir = bool(lane & width);
                Bitonic<1>::merge_impl(dir, width, &keys[2], &indices[2]);
            }
            Bitonic<1>::merge_impl(false, warp_width, &keys[2], &indices[2]);
            
            /* bitonic<1>::sort_impl(true, ..., keys[3]) */
            for (int width = 2; width < warp_width; width <<= 1) {
                bool dir = bool(lane & width);
                Bitonic<1>::merge_impl(dir, width, &keys[3], &indices[3]);
            }
            Bitonic<1>::merge_impl(true, warp_width, &keys[3], &indices[3]);
            
            /* bitonic<2>::merge_impl(true) - 完成 bitonic<2>::sort_impl 的最后一步 */
            Bitonic<2>::merge_impl(true, warp_width, &keys[2], &indices[2]);
            
            /* === 第三步：最后的 merge，合并成完整的有序序列 === */
            /* 注意：这里不能直接用 merge_impl(ascending, warp_width, keys, indices)
             * 因为那会调用 Bitonic<4>::merge_impl，我们需要显式调用它 */
        } else if constexpr (Size == 8) {
            /* 
             * Size == 8: 手动展开递归逻辑
             * 
             * RAFT 递归逻辑：
             *   bitonic<4>::sort_impl(false, warp_width, keys);      // keys[0:4] 降序
             *   bitonic<4>::sort_impl(true, warp_width, keys+4);     // keys[4:8] 升序
             *   bitonic<8>::merge_impl(ascending, warp_width, keys); // 最后merge
             * 
             * 由于 bitonic<4>::sort_impl 本身很复杂，这里需要完整展开它
             */
            const int lane = laneId();
            
            /* === 第一步：对 keys[0:4] 按降序排序（展开 bitonic<4>::sort_impl(false, ...)）=== */
            
            // bitonic<2>::sort_impl(false, ..., keys[0:2])
            for (int width = 2; width < warp_width; width <<= 1) {
                Bitonic<1>::merge_impl(bool(lane & width), width, &keys[0], &indices[0]);
            }
            Bitonic<1>::merge_impl(false, warp_width, &keys[0], &indices[0]);
            
            for (int width = 2; width < warp_width; width <<= 1) {
                Bitonic<1>::merge_impl(bool(lane & width), width, &keys[1], &indices[1]);
            }
            Bitonic<1>::merge_impl(true, warp_width, &keys[1], &indices[1]);
            
            Bitonic<2>::merge_impl(false, warp_width, &keys[0], &indices[0]);
            
            // bitonic<2>::sort_impl(true, ..., keys[2:4])
            for (int width = 2; width < warp_width; width <<= 1) {
                Bitonic<1>::merge_impl(bool(lane & width), width, &keys[2], &indices[2]);
            }
            Bitonic<1>::merge_impl(false, warp_width, &keys[2], &indices[2]);
            
            for (int width = 2; width < warp_width; width <<= 1) {
                Bitonic<1>::merge_impl(bool(lane & width), width, &keys[3], &indices[3]);
            }
            Bitonic<1>::merge_impl(true, warp_width, &keys[3], &indices[3]);
            
            Bitonic<2>::merge_impl(true, warp_width, &keys[2], &indices[2]);
            
            // bitonic<4>::merge_impl(false) - 完成 bitonic<4>::sort_impl 的最后一步
            Bitonic<4>::merge_impl(false, warp_width, &keys[0], &indices[0]);
            
            /* === 第二步：对 keys[4:8] 按升序排序（展开 bitonic<4>::sort_impl(true, ...)）=== */
            
            // bitonic<2>::sort_impl(false, ..., keys[4:6])
            for (int width = 2; width < warp_width; width <<= 1) {
                Bitonic<1>::merge_impl(bool(lane & width), width, &keys[4], &indices[4]);
            }
            Bitonic<1>::merge_impl(false, warp_width, &keys[4], &indices[4]);
            
            for (int width = 2; width < warp_width; width <<= 1) {
                Bitonic<1>::merge_impl(bool(lane & width), width, &keys[5], &indices[5]);
            }
            Bitonic<1>::merge_impl(true, warp_width, &keys[5], &indices[5]);
            
            Bitonic<2>::merge_impl(false, warp_width, &keys[4], &indices[4]);
            
            // bitonic<2>::sort_impl(true, ..., keys[6:8])
            for (int width = 2; width < warp_width; width <<= 1) {
                Bitonic<1>::merge_impl(bool(lane & width), width, &keys[6], &indices[6]);
            }
            Bitonic<1>::merge_impl(false, warp_width, &keys[6], &indices[6]);
            
            for (int width = 2; width < warp_width; width <<= 1) {
                Bitonic<1>::merge_impl(bool(lane & width), width, &keys[7], &indices[7]);
            }
            Bitonic<1>::merge_impl(true, warp_width, &keys[7], &indices[7]);
            
            Bitonic<2>::merge_impl(true, warp_width, &keys[6], &indices[6]);
            
            // bitonic<4>::merge_impl(true) - 完成 bitonic<4>::sort_impl 的最后一步
            Bitonic<4>::merge_impl(true, warp_width, &keys[4], &indices[4]);
        }
        
        /* 最后统一 merge */
        merge_impl(ascending, warp_width, keys, indices);
    }
    
    template<typename KeyT, typename IdxT>
    static __device__ __forceinline__ void compare_and_swap(
        bool ascending, KeyT& a, KeyT& b, IdxT& ia, IdxT& ib)
    {
        bool cond = ascending ? (a > b) : (a < b);
        if (cond) {
            swap(a, b);
            swap(ia, ib);
        }
    }
};

} // namespace bitonic
} // namespace pgvector

#endif // BITONIC_CUH

