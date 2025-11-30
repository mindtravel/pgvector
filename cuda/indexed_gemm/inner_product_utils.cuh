#ifndef INNER_PRODUCT_UTILS_CUH
#define INNER_PRODUCT_UTILS_CUH

#include "../pch.h"
#include <stdint.h>

/**
 * 内积计算工具函数
 * 
 * 这些函数用于优化向量内积计算，支持：
 * - float4向量化加载和计算
 * - 对齐和未对齐内存访问
 * - 编译时和运行时维度
 */

/**
 * 加载tile数据（float4向量化）
 * 
 * @tparam Tile tile大小
 * @param lhs_vec4 左侧向量的float4指针
 * @param rhs_vec4 右侧向量的float4指针
 * @param base_idx 起始索引
 * @param vec4_count 总float4数量
 * @param lhs_tile 输出的左侧tile
 * @param rhs_tile 输出的右侧tile
 */
template<int Tile>
__device__ __forceinline__ void load_tile_vec4(const float4* lhs_vec4,
                                               const float4* rhs_vec4,
                                               int base_idx,
                                               int vec4_count,
                                               float4 (&lhs_tile)[Tile],
                                               float4 (&rhs_tile)[Tile]) {
    #pragma unroll
    for (int t = 0; t < Tile; ++t) {
        int idx = base_idx + t;
        if (idx < vec4_count) {
            lhs_tile[t] = lhs_vec4[idx];
            rhs_tile[t] = rhs_vec4[idx];
        } else {
            lhs_tile[t] = make_float4(0.f, 0.f, 0.f, 0.f);
            rhs_tile[t] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
}

/**
 * 累加tile数据（float4向量化）
 * 
 * @tparam Tile tile大小
 * @param lhs_tile 左侧tile
 * @param rhs_tile 右侧tile
 * @param valid_count 有效元素数量
 * @param sum 累加和
 * @return 累加后的和
 */
template<int Tile>
__device__ __forceinline__ float accumulate_tile(const float4 (&lhs_tile)[Tile],
                                                 const float4 (&rhs_tile)[Tile],
                                                 int valid_count,
                                                 float sum) {
    #pragma unroll
    for (int t = 0; t < Tile; ++t) {
        if (t < valid_count) {
            const float4& l = lhs_tile[t];
            const float4& r = rhs_tile[t];
            sum = fmaf(l.x, r.x, sum);
            sum = fmaf(l.y, r.y, sum);
            sum = fmaf(l.z, r.z, sum);
            sum = fmaf(l.w, r.w, sum);
        }
    }
    return sum;
}

/**
 * 编译时维度优化的内积计算（tiled版本）
 * 
 * @tparam Dim 向量维度（编译时常量）
 * @param lhs 左侧向量
 * @param rhs 右侧向量
 * @return 内积结果
 */
template<int Dim>
__device__ __forceinline__ float dot_product_tiled(const float* __restrict__ lhs,
                                                   const float* __restrict__ rhs) {
    constexpr int kVec4Count = Dim / 4;
    constexpr int kTile = 4;
    if constexpr (kVec4Count == 0) {
        return 0.0f;
    } else {
        constexpr int tile_count = (kVec4Count + kTile - 1) / kTile;
        const float4* lhs_vec4 = reinterpret_cast<const float4*>(lhs);
        const float4* rhs_vec4 = reinterpret_cast<const float4*>(rhs);

        float4 cur_lhs[kTile];
        float4 cur_rhs[kTile];
        load_tile_vec4<kTile>(lhs_vec4, rhs_vec4, 0, kVec4Count, cur_lhs, cur_rhs);

        float sum = 0.0f;

        if constexpr (tile_count == 1) {
            sum = accumulate_tile<kTile>(cur_lhs, cur_rhs, kVec4Count, sum);
        } else {
            float4 next_lhs[kTile];
            float4 next_rhs[kTile];

            #pragma unroll
            for (int tile = 0; tile < tile_count; ++tile) {
                int next_base = (tile + 1) * kTile;
                if (tile + 1 < tile_count) {
                    load_tile_vec4<kTile>(lhs_vec4, rhs_vec4, next_base, kVec4Count, next_lhs, next_rhs);
                }

                int valid = kVec4Count - tile * kTile;
                valid = valid > kTile ? kTile : valid;
                sum   = accumulate_tile<kTile>(cur_lhs, cur_rhs, valid, sum);

                if (tile + 1 < tile_count) {
                    #pragma unroll
                    for (int t = 0; t < kTile; ++t) {
                        cur_lhs[t] = next_lhs[t];
                        cur_rhs[t] = next_rhs[t];
                    }
                }
            }
        }

        return sum;
    }
}

/**
 * 对齐内存的内积计算（float4向量化）
 * 
 * @param lhs 左侧向量（必须对齐到16字节）
 * @param rhs 右侧向量（必须对齐到16字节）
 * @param length 向量长度（必须是4的倍数）
 * @return 内积结果
 */
__device__ __forceinline__ float dot_product_vec4_aligned(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    int length) {
    float sum = 0.0f;
    const int vec4_elems = length >> 2;
    const float4* lhs_vec4 = reinterpret_cast<const float4*>(lhs);
    const float4* rhs_vec4 = reinterpret_cast<const float4*>(rhs);
    
    #pragma unroll
    for (int v = 0; v < vec4_elems; ++v) {
        const float4 lhs_val = lhs_vec4[v];
        const float4 rhs_val = rhs_vec4[v];
        sum += lhs_val.x * rhs_val.x +
               lhs_val.y * rhs_val.y +
               lhs_val.z * rhs_val.z +
               lhs_val.w * rhs_val.w;
    }
    return sum;
}

/**
 * 通用内积计算（支持未对齐内存）
 * 
 * 自动处理对齐和未对齐的内存访问：
 * 1. 先处理未对齐的前缀部分（标量）
 * 2. 处理对齐的中间部分（float4向量化）
 * 3. 处理剩余的后缀部分（标量）
 * 
 * @param lhs 左侧向量
 * @param rhs 右侧向量
 * @param length 向量长度
 * @return 内积结果
 */
__device__ __forceinline__ float dot_product_accumulate(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    int length) {
    float sum = 0.0f;
    int i = 0;
    
    while (i < length &&
           ((reinterpret_cast<uintptr_t>(lhs + i) |
             reinterpret_cast<uintptr_t>(rhs + i)) & (sizeof(float4) - 1))) {
        sum += lhs[i] * rhs[i];
        ++i;
    }
    
    const int remaining = length - i;
    const int vec4_elems = remaining >> 2;
    
    if (vec4_elems > 0) {
        const float4* lhs_vec4 = reinterpret_cast<const float4*>(lhs + i);
        const float4* rhs_vec4 = reinterpret_cast<const float4*>(rhs + i);
        
        #pragma unroll
        for (int v = 0; v < vec4_elems; ++v) {
            const float4 lhs_val = lhs_vec4[v];
            const float4 rhs_val = rhs_vec4[v];
            sum += lhs_val.x * rhs_val.x +
                   lhs_val.y * rhs_val.y +
                   lhs_val.z * rhs_val.z +
                   lhs_val.w * rhs_val.w;
        }
        i += vec4_elems << 2;
    }
    
    for (; i < length; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

#endif // INNER_PRODUCT_UTILS_CUH

