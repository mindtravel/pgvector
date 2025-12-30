// kmeans_reorder.cu
#include "kmeans.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>

static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
        std::abort();
    }
}

static inline void cuda_check_last(const char* msg) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error (%s): %s\n", msg, cudaGetErrorString(e));
        std::abort();
    }
}

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

// ============================================================
// Public API
// ============================================================
//
// Build permutation perm[0..n-1] such that points belonging to cluster 0 come first,
// then cluster 1, etc. (order within each cluster is arbitrary, but stable enough).
//
// Memory:
// - perm is allocated on device inside this function and copied back to host if requested.
// - To avoid giant pinned output, we return perm and cluster_info.
// - Caller can produce reordered vectors by: out[p] = in[perm[p]].
//
// Note: This routine streams assign in batches to reduce host->device bandwidth pressure,
// and does NOT require input vectors at all.
//
// ============================================================
void gpu_build_permutation_by_cluster(
    const KMeansCase& cfg,
    const int* h_assign,             // [n] host
    int* h_perm_out,                 // [n] host (optional, can be nullptr)
    ClusterInfo* h_cluster_info,      // optional host output
    int device_id,
    int B,                           // e.g. 1<<20
    cudaStream_t stream              // can pass 0
) {
    const int n = cfg.n;
    const int k = cfg.k;
    if (n <= 0 || k <= 0) return;

    cuda_check(cudaSetDevice(device_id), "set device");

    // Device buffers (O(k) + O(B) + O(n int))
    int* d_counts = nullptr;                         // [k]
    unsigned long long* d_writeptr = nullptr;        // [k]
    int* d_assign_b[2] = {nullptr, nullptr};         // [B]
    unsigned long long* d_pos_b[2] = {nullptr, nullptr}; // [B]
    int* d_perm = nullptr;                           // [n] permutation indices

    unsigned int* d_invalid = nullptr;
    unsigned int* d_oob = nullptr;

    cuda_check(cudaMalloc(&d_counts, sizeof(int) * (size_t)k), "malloc d_counts");
    cuda_check(cudaMalloc(&d_writeptr, sizeof(unsigned long long) * (size_t)k), "malloc d_writeptr");
    cuda_check(cudaMalloc(&d_perm, sizeof(int) * (size_t)n), "malloc d_perm");

    cuda_check(cudaMalloc(&d_invalid, sizeof(unsigned int)), "malloc d_invalid");
    cuda_check(cudaMalloc(&d_oob, sizeof(unsigned int)), "malloc d_oob");
    cuda_check(cudaMemsetAsync(d_invalid, 0, sizeof(unsigned int), stream), "memset invalid");
    cuda_check(cudaMemsetAsync(d_oob, 0, sizeof(unsigned int), stream), "memset oob");

    for (int t = 0; t < 2; ++t) {
        cuda_check(cudaMalloc(&d_assign_b[t], sizeof(int) * (size_t)B), "malloc d_assign_b");
        cuda_check(cudaMalloc(&d_pos_b[t], sizeof(unsigned long long) * (size_t)B), "malloc d_pos_b");
    }

    // ------------------------------------------------------------
    // PASS 1: count clusters (streaming assign only)
    // ------------------------------------------------------------
    cuda_check(cudaMemsetAsync(d_counts, 0, sizeof(int) * (size_t)k, stream), "memset counts");

    int cur_buf = 0;
    for (int base = 0; base < n; base += B) {
        int curB = std::min(B, n - base);
        int buf = cur_buf;

        // H2D assign batch
        cuda_check(cudaMemcpyAsync(d_assign_b[buf], h_assign + base,
                                   sizeof(int) * (size_t)curB,
                                   cudaMemcpyHostToDevice, stream),
                   "h2d assign batch");

        // Count
        int threads = 256;
        int blocks = (curB + threads - 1) / threads;
        kernel_count_clusters<<<blocks, threads, 0, stream>>>(
            d_assign_b[buf], d_counts, curB, k, d_invalid);
        cuda_check_last("kernel_count_clusters");

        cur_buf = 1 - cur_buf;
    }

    cuda_check(cudaStreamSynchronize(stream), "sync after pass1");

    // ------------------------------------------------------------
    // Build offsets on host
    // ------------------------------------------------------------
    std::vector<int> h_counts(k);
    cuda_check(cudaMemcpy(h_counts.data(), d_counts, sizeof(int) * (size_t)k,
                          cudaMemcpyDeviceToHost),
               "d2h counts");

    std::vector<unsigned long long> h_offsets;
    unsigned long long total = 0ULL;
    build_offsets_host(h_counts, h_offsets, total);

    if ((long long)total != (long long)n) {
        fprintf(stderr,
                "[reorder] WARNING: total counts (%llu) != n (%d). "
                "Some assignments are invalid/out-of-range.\n",
                (unsigned long long)total, n);
    }

    // Init write_ptr = offsets
    cuda_check(cudaMemcpy(d_writeptr, h_offsets.data(),
                          sizeof(unsigned long long) * (size_t)k,
                          cudaMemcpyHostToDevice),
               "h2d writeptr offsets");

    // Optional: output cluster info
    if (h_cluster_info) {
        h_cluster_info->k = k;
        h_cluster_info->counts = (int*)std::malloc(sizeof(int) * (size_t)k);
        h_cluster_info->offsets = (long long*)std::malloc(sizeof(long long) * (size_t)k);
        if (!h_cluster_info->counts || !h_cluster_info->offsets) {
            fprintf(stderr, "[reorder] ERROR: malloc failed for cluster_info.\n");
            std::abort();
        }
        std::memcpy(h_cluster_info->counts, h_counts.data(), sizeof(int) * (size_t)k);
        // store offsets in signed long long for your existing struct
        for (int c = 0; c < k; ++c) h_cluster_info->offsets[c] = (long long)h_offsets[c];
    }

    // ------------------------------------------------------------
    // PASS 2: compute positions + scatter permutation (streaming)
    // ------------------------------------------------------------
    cur_buf = 0;
    for (int base = 0; base < n; base += B) {
        int curB = std::min(B, n - base);
        int buf = cur_buf;

        cuda_check(cudaMemcpyAsync(d_assign_b[buf], h_assign + base,
                                   sizeof(int) * (size_t)curB,
                                   cudaMemcpyHostToDevice, stream),
                   "pass2 h2d assign");

        int threads = 256;
        int blocks = (curB + threads - 1) / threads;

        // Compute positions (batch-local pos)
        kernel_compute_positions_u64<<<blocks, threads, 0, stream>>>(
            d_assign_b[buf], d_writeptr, d_pos_b[buf], curB, k, d_invalid);
        cuda_check_last("kernel_compute_positions_u64");

        // Scatter indices into perm[p]
        kernel_scatter_perm<<<blocks, threads, 0, stream>>>(
            d_pos_b[buf], d_perm, base, curB, (unsigned long long)n, d_oob);
        cuda_check_last("kernel_scatter_perm");

        cur_buf = 1 - cur_buf;
    }

    cuda_check(cudaStreamSynchronize(stream), "sync after pass2");

    // Check invalid/oob counters
    unsigned int h_invalid = 0, h_oob = 0;
    cuda_check(cudaMemcpy(&h_invalid, d_invalid, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost),
               "d2h invalid");
    cuda_check(cudaMemcpy(&h_oob, d_oob, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost),
               "d2h oob");
    if (h_invalid != 0 || h_oob != 0) {
        fprintf(stderr, "[reorder] WARNING: invalid_assign=%u, oob_scatter=%u\n",
                h_invalid, h_oob);
    }

    // Copy perm back if requested
    if (h_perm_out) {
        cuda_check(cudaMemcpy(h_perm_out, d_perm, sizeof(int) * (size_t)n,
                              cudaMemcpyDeviceToHost),
                   "d2h perm");
    }

    // Cleanup
    for (int t = 0; t < 2; ++t) {
        cudaFree(d_assign_b[t]);
        cudaFree(d_pos_b[t]);
    }
    cudaFree(d_counts);
    cudaFree(d_writeptr);
    cudaFree(d_perm);
    cudaFree(d_invalid);
    cudaFree(d_oob);
}

// ============================================================
// CPU-side reordering: build reordered vectors from permutation
// ============================================================

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

// ============================================================
// IVF K-means: Complete pipeline from clustering to reordering
// ============================================================

bool ivf_kmeans(
    const KMeansCase& cfg,
    const float* h_data_in,        // [n, dim] CPU input
    float* h_data_out,             // [n, dim] CPU output (must be pre-allocated)
    float* d_centroids,            // [k, dim] GPU (in/out)
    ClusterInfo* h_cluster_info,   // optional output
    bool use_minibatch,            // true for minibatch, false for Lloyd
    int device_id,
    int batch_size,
    float* h_objective             // optional output
) {
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;
    
    // 参数验证
    if (!h_data_in || !h_data_out || !d_centroids || n <= 0 || dim <= 0 || k <= 0) {
        fprintf(stderr, "[ivf_kmeans] ERROR: Invalid parameters\n");
        return false;
    }
    
    cuda_check(cudaSetDevice(device_id), "set device");
    
    // Step 1: Allocate device buffer for assignments
    int* d_assign = nullptr;
    cuda_check(cudaMalloc(&d_assign, sizeof(int) * (size_t)n), "malloc d_assign");
    
    // Step 2: Run K-means clustering
    float obj = 0.0f;
    if (use_minibatch) {
        gpu_kmeans_minibatch(cfg, h_data_in, d_assign, d_centroids, &obj);
    } else {
        gpu_kmeans_lloyd(cfg, h_data_in, d_assign, d_centroids, &obj);
    }
    cuda_check(cudaDeviceSynchronize(), "sync after kmeans");
    cuda_check_last("kmeans kernels");
    
    if (h_objective) {
        *h_objective = obj;
    }
    
    // Step 3: Copy assign back to host (needed for permutation building)
    std::vector<int> h_assign(n);
    cuda_check(cudaMemcpy(h_assign.data(), d_assign, sizeof(int) * (size_t)n,
                         cudaMemcpyDeviceToHost), "d2h assign");
    
    // Step 4: Build permutation on GPU
    std::vector<int> h_perm(n);
    ClusterInfo cluster_info;
    ClusterInfo* info_ptr = h_cluster_info ? h_cluster_info : &cluster_info;
    
    gpu_build_permutation_by_cluster(cfg, h_assign.data(), h_perm.data(), info_ptr,
                                    device_id, batch_size, 0);
    cuda_check(cudaDeviceSynchronize(), "sync after build_permutation");
    cuda_check_last("build_permutation");
    
    // Step 5: Reorder vectors on CPU
    cpu_reorder_vectors_by_permutation(h_data_in, h_perm.data(), h_data_out, n, dim);
    
    // Cleanup
    cudaFree(d_assign);
    
    // Only free cluster_info if we created a temporary one
    if (!h_cluster_info) {
        free_cluster_info(&cluster_info, false);
    }
    
    return true;
}

// Keep your existing free_cluster_info()
void free_cluster_info(ClusterInfo* info, bool is_device) {
    if (!info) return;
    if (is_device) {
        if (info->offsets) cudaFree(info->offsets);
        if (info->counts) cudaFree(info->counts);
    } else {
        if (info->offsets) std::free(info->offsets);
        if (info->counts) std::free(info->counts);
    }
    info->offsets = nullptr;
    info->counts = nullptr;
    info->k = 0;
}
