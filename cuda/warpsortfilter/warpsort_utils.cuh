/**
 * Warp-Sort Utility Functions and Constants for pgvector
 * 
 * This file contains common constants and device utility functions
 * used by warp-sort based algorithms.
 * 
 * Copyright (c) 2024, pgvector
 * Adapted from RAFT (Apache 2.0 License)
 */

#ifndef WARPSORT_UTILS_CUH
#define WARPSORT_UTILS_CUH

#include <limits>
#include <type_traits>
#include <math_constants.h>

namespace pgvector {
namespace warpsort_utils {

// ============================================================================
// Constants and Configuration
// ============================================================================

static constexpr int kMaxCapacity = 256;
static constexpr int kWarpSize = 32;

// ============================================================================
// Device Utility Functions
// ============================================================================

/** Get lane ID within a warp */
__device__ __forceinline__ int laneId() 
{
    return threadIdx.x % kWarpSize;
}

/** Warp-level any() predicate */
__device__ __forceinline__ bool any(bool predicate)
{
    return __any_sync(0xffffffff, predicate);
}

/** Warp shuffle operation */
template<typename T>
__device__ __forceinline__ T shfl(T var, int srcLane, int width = kWarpSize)
{
    return __shfl_sync(0xffffffff, var, srcLane, width);
}

/** Swap two values */
template<typename T>
__device__ __forceinline__ void swap(T& a, T& b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

/** Check if value is power of 2 */
constexpr bool isPowerOf2(int n) 
{
    return n > 0 && (n & (n - 1)) == 0;
}

/** Upper/Lower bounds for comparison */
template<typename T>
__device__ __host__ __forceinline__ T upper_bound()
{
    return std::numeric_limits<T>::max();
}

template<typename T>
__device__ __host__ __forceinline__ T lower_bound()
{
    return std::numeric_limits<T>::lowest();
}

/* 特化 for float - 使用无穷大作为哨兵值 */
template<>
__device__ __host__ __forceinline__ float upper_bound<float>()
{
    return INFINITY;  /* 正无穷 */
}

template<>
__device__ __host__ __forceinline__ float lower_bound<float>()
{
    return -INFINITY;  /* 负无穷 */
}

/** Comparison helper */
template<bool Ascending, typename T>
__device__ __forceinline__ bool is_ordered(T left, T right)
{
    if constexpr (Ascending) {
        return left < right;
    } else {
        return left > right;
    }
}

// ============================================================================
// Power-of-2 Utilities
// ============================================================================

template<int N> /*这里N是2的幂*/
struct Pow2 {
    static_assert(isPowerOf2(N), "N must be power of 2");
    static constexpr int Log2 = __builtin_ctz(N);
    static constexpr int Mask = N - 1;
    
    __device__ __forceinline__ static int mod(int x) { return x & Mask; }
    __device__ __forceinline__ static int div(int x) { return x >> Log2; }
    __device__ __forceinline__ static int roundDown(int x) { return x & ~Mask; }
    __device__ __forceinline__ static int roundUp(int x) { return (x + Mask) & ~Mask; }
};


} // namespace warpsort_utils
} // namespace pgvector

#endif // WARPSORT_UTILS_CUH

