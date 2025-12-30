// kmeans_reorder_utils.cu
#include "kmeans_reorder_utils.cuh"
#include <cstring>
#include <thread>
#include <algorithm>

// ============================================================
// GPU Kernels Implementation
// ============================================================

// Pass1: count cluster sizes
__global__ void kernel_count_clusters(const int* __restrict__ assign,
                                      int* __restrict__ counts,
                                      int n, int k,
                                      unsigned int* __restrict__ invalid_counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int c = assign[i];
        if (c >= 0 && c < k) {
            atomicAdd(&counts[c], 1);
        } else {
            if (invalid_counter) atomicAdd(invalid_counter, 1u);
        }
    }
}

// Pass2: compute output positions per element by atomicAdd on write_ptr (uint64)
__global__ void kernel_compute_positions_u64(const int* __restrict__ assign,
                                            unsigned long long* __restrict__ write_ptr, // [k]
                                            unsigned long long* __restrict__ out_pos,   // [n] (batch-local or global)
                                            int n, int k,
                                            unsigned int* __restrict__ invalid_counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int c = assign[i];
        if (c >= 0 && c < k) {
            unsigned long long p = atomicAdd(&write_ptr[c], 1ULL);
            out_pos[i] = p;
        } else {
            out_pos[i] = ~0ULL; // invalid
            if (invalid_counter) atomicAdd(invalid_counter, 1u);
        }
    }
}

// Pass3: scatter global indices into permutation array:
// perm[p] = global_index
__global__ void kernel_scatter_perm(const unsigned long long* __restrict__ pos, // [curB]
                                   int* __restrict__ perm,                     // [n]
                                   int base, int curB, unsigned long long n,
                                   unsigned int* __restrict__ oob_counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < curB) {
        unsigned long long p = pos[i];
        if (p < n) {
            perm[p] = base + i;
        } else {
            if (oob_counter) atomicAdd(oob_counter, 1u);
        }
    }
}

// ============================================================
// Host-side Utility Functions Implementation
// ============================================================

/**
 * 根据permutation数组重排向量（多线程CPU实现）
 */
void cpu_reorder_vectors_by_permutation(
    const float* h_data_in,    // [n, dim] CPU
    const int* h_perm,         // [n] CPU permutation array
    float* h_data_out,         // [n, dim] CPU output
    int n, int dim
) {
    if (!h_data_in || !h_perm || !h_data_out || n <= 0 || dim <= 0) {
        fprintf(stderr, "[reorder] ERROR: Invalid parameters for cpu_reorder_vectors_by_permutation\n");
        return;
    }
    
    // 多线程重排：out[p] = in[perm[p]]
    const int num_threads = std::max(1u, std::thread::hardware_concurrency());
    const int chunk = (n + num_threads - 1) / num_threads;
    std::vector<std::thread> th;
    th.reserve(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        th.emplace_back([=]() {
            int p0 = t * chunk;
            int p1 = std::min(n, p0 + chunk);
            for (int p = p0; p < p1; ++p) {
                int i = h_perm[p];
                // 检查索引有效性（perm中可能存在无效值）
                if (i < 0 || i >= n) continue;
                const float* src = h_data_in + (size_t)i * dim;
                float* dst = h_data_out + (size_t)p * dim;
                std::memcpy(dst, src, sizeof(float) * (size_t)dim);
            }
        });
    }
    
    for (auto& x : th) x.join();
}

/**
 * 根据permutation数组重排索引（多线程CPU实现）
 */
void cpu_reorder_indices_by_permutation(
    const int* h_indices_in,   // [n] CPU 原始索引数组
    const int* h_perm,         // [n] CPU permutation array
    int* h_indices_out,        // [n] CPU 重排后的索引数组
    int n
) {
    if (!h_indices_in || !h_perm || !h_indices_out || n <= 0) {
        fprintf(stderr, "[reorder] ERROR: Invalid parameters for cpu_reorder_indices_by_permutation\n");
        return;
    }
    
    // 多线程重排：out[p] = in[perm[p]]
    const int num_threads = std::max(1u, std::thread::hardware_concurrency());
    const int chunk = (n + num_threads - 1) / num_threads;
    std::vector<std::thread> th;
    th.reserve(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        th.emplace_back([=]() {
            int p0 = t * chunk;
            int p1 = std::min(n, p0 + chunk);
            for (int p = p0; p < p1; ++p) {
                int i = h_perm[p];
                // 检查索引有效性（perm中可能存在无效值）
                if (i < 0 || i >= n) continue;
                h_indices_out[p] = h_indices_in[i];
            }
        });
    }
    
    for (auto& x : th) x.join();
}


