#ifndef KMEANS_REORDER_UTILS_CUH
#define KMEANS_REORDER_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>

// ============================================================
// Utility Functions
// ============================================================

/**
 * CUDA错误检查工具函数
 */
static inline void cuda_check_reorder(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
        std::abort();
    }
}

/**
 * CUDA kernel错误检查工具函数
 */
static inline void cuda_check_last_reorder(const char* msg) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error (%s): %s\n", msg, cudaGetErrorString(e));
        std::abort();
    }
}

// ============================================================
// GPU Kernels
// ============================================================

/**
 * Pass1: 统计每个cluster的向量数量
 * 
 * @param assign 输入：每个向量的cluster分配 [n]
 * @param counts 输出：每个cluster的向量数量 [k]
 * @param n 向量总数
 * @param k cluster数量
 * @param invalid_counter 输出：无效分配的计数器（可选）
 */
__global__ void kernel_count_clusters(const int* __restrict__ assign,
                                      int* __restrict__ counts,
                                      int n, int k,
                                      unsigned int* __restrict__ invalid_counter);

/**
 * Pass2: 计算每个向量在重排后的输出位置（使用uint64原子操作）
 * 
 * @param assign 输入：每个向量的cluster分配 [n]
 * @param write_ptr 输入输出：每个cluster的写入指针 [k]
 * @param out_pos 输出：每个向量的输出位置 [n]（batch-local或global）
 * @param n 向量总数
 * @param k cluster数量
 * @param invalid_counter 输出：无效分配的计数器（可选）
 */
__global__ void kernel_compute_positions_u64(const int* __restrict__ assign,
                                            unsigned long long* __restrict__ write_ptr, // [k]
                                            unsigned long long* __restrict__ out_pos,   // [n] (batch-local or global)
                                            int n, int k,
                                            unsigned int* __restrict__ invalid_counter);

/**
 * Pass3: 将全局索引scatter到permutation数组中
 * perm[p] = global_index
 * 
 * @param pos 输入：每个向量的位置 [curB]
 * @param perm 输出：permutation数组 [n]
 * @param base 当前batch的起始索引
 * @param curB 当前batch的大小
 * @param n 总向量数
 * @param oob_counter 输出：越界计数器（可选）
 */
__global__ void kernel_scatter_perm(const unsigned long long* __restrict__ pos, // [curB]
                                   int* __restrict__ perm,                     // [n]
                                   int base, int curB, unsigned long long n,
                                   unsigned int* __restrict__ oob_counter);

// ============================================================
// Host-side Utility Functions
// ============================================================

/**
 * 在host端构建offsets数组
 * 
 * @param counts 输入：每个cluster的向量数量
 * @param offsets 输出：每个cluster的起始偏移量
 * @param total 输出：总向量数
 */
static inline void build_offsets_host(const std::vector<int>& counts,
                                      std::vector<unsigned long long>& offsets,
                                      unsigned long long& total) {
    const int k = (int)counts.size();
    offsets.resize(k);
    total = 0ULL;
    for (int c = 0; c < k; ++c) {
        offsets[c] = total;
        total += (unsigned long long)counts[c];
    }
}

/**
 * 根据permutation数组重排向量（多线程CPU实现）
 * 
 * 使用多线程并行处理，提高性能
 * 使用pageable memory，避免pinned memory限制
 * 
 * @param h_data_in 输入：原始向量数据 [n, dim] row-major
 * @param h_perm 输入：permutation数组 [n]，perm[p] 表示重排后位置p对应的原始索引
 * @param h_data_out 输出：重排后的向量数据 [n, dim] row-major
 * @param n 向量数量
 * @param dim 向量维度
 */
void cpu_reorder_vectors_by_permutation(
    const float* h_data_in,    // [n, dim] CPU
    const int* h_perm,         // [n] CPU permutation array
    float* h_data_out,         // [n, dim] CPU output
    int n, int dim
);

/**
 * 根据permutation数组重排索引（多线程CPU实现）
 * 
 * 重排规则：out[p] = in[perm[p]]
 * 例如：如果 perm[0]=5, perm[1]=2，则 out[0]=in[5], out[1]=in[2]
 * 
 * @param h_indices_in 输入：原始索引数组 [n]，通常是 [0, 1, 2, ..., n-1]
 * @param h_perm 输入：permutation数组 [n]，perm[p] 表示重排后位置p对应的原始索引
 * @param h_indices_out 输出：重排后的索引数组 [n]
 * @param n 向量数量
 */
void cpu_reorder_indices_by_permutation(
    const int* h_indices_in,   // [n] CPU 原始索引数组
    const int* h_perm,         // [n] CPU permutation array
    int* h_indices_out,        // [n] CPU 重排后的索引数组
    int n
);

#endif // KMEANS_REORDER_UTILS_CUH


