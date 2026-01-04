// kmeans_reorder.cu
#include "kmeans.cuh"
#include "kmeans_reorder_utils.cuh"
#include "../utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>

// ============================================================
// GPU版本：构建permutation数组（设备指针版本，避免H2D传输）
// ============================================================

void gpu_build_permutation_by_cluster(
    const KMeansCase& cfg,
    const int* d_assign,             // [n] device
    int* h_perm_out,                 // [n] host (optional, can be nullptr)
    ClusterInfo* h_cluster_info,      // optional host output
    int device_id,
    int B,                           // e.g. 1<<20
    cudaStream_t stream              // can pass 0 (default in header)
) {
    const int n = cfg.n;
    const int k = cfg.k;
    if (n <= 0 || k <= 0) return;

    cudaSetDevice(device_id);
    CHECK_CUDA_ERRORS;

    // Device buffers (O(k) + O(B) + O(n int))
    int* d_counts = nullptr;                         // [k]
    unsigned long long* d_writeptr = nullptr;        // [k]
    unsigned long long* d_pos_b[2] = {nullptr, nullptr}; // [B]
    int* d_perm = nullptr;                           // [n] permutation indices

    unsigned int* d_invalid = nullptr;
    unsigned int* d_oob = nullptr;

    cudaMalloc(&d_counts, sizeof(int) * (size_t)k);
    cudaMalloc(&d_writeptr, sizeof(unsigned long long) * (size_t)k);
    cudaMalloc(&d_perm, sizeof(int) * (size_t)n);
    CHECK_CUDA_ERRORS;

    cudaMalloc(&d_invalid, sizeof(unsigned int));
    cudaMalloc(&d_oob, sizeof(unsigned int));
    cudaMemsetAsync(d_invalid, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(d_oob, 0, sizeof(unsigned int), stream);
    CHECK_CUDA_ERRORS;

    for (int t = 0; t < 2; ++t) {
        cudaMalloc(&d_pos_b[t], sizeof(unsigned long long) * (size_t)B);
    }
    CHECK_CUDA_ERRORS;

    // ------------------------------------------------------------
    // PASS 1: count clusters (streaming assign directly from device)
    // ------------------------------------------------------------
    cudaMemsetAsync(d_counts, 0, sizeof(int) * (size_t)k, stream);
    CHECK_CUDA_ERRORS;

    int cur_buf = 0;
    for (int base = 0; base < n; base += B) {
        int curB = std::min(B, n - base);
        int buf = cur_buf;

        // 直接使用设备指针，不需要 H2D 拷贝
        const int* d_assign_batch = d_assign + base;

        // Count
        int threads = 256;
        int blocks = (curB + threads - 1) / threads;
        kernel_count_clusters<<<blocks, threads, 0, stream>>>(
            d_assign_batch, d_counts, curB, k, d_invalid);
        CHECK_CUDA_ERRORS;

        cur_buf = 1 - cur_buf;
    }

    cudaStreamSynchronize(stream);
    CHECK_CUDA_ERRORS;

    // ------------------------------------------------------------
    // Build offsets on host
    // ------------------------------------------------------------
    std::vector<int> h_counts(k);
    cudaMemcpy(h_counts.data(), d_counts, sizeof(int) * (size_t)k,
               cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;

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
    cudaMemcpy(d_writeptr, h_offsets.data(),
               sizeof(unsigned long long) * (size_t)k,
               cudaMemcpyHostToDevice);
    CHECK_CUDA_ERRORS;

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

        // 直接使用设备指针，不需要 H2D 拷贝
        const int* d_assign_batch = d_assign + base;

        int threads = 256;
        int blocks = (curB + threads - 1) / threads;

        // Compute positions (batch-local pos)
        kernel_compute_positions_u64<<<blocks, threads, 0, stream>>>(
            d_assign_batch, d_writeptr, d_pos_b[buf], curB, k, d_invalid);
        CHECK_CUDA_ERRORS;

        // Scatter indices into perm[p]
        kernel_scatter_perm<<<blocks, threads, 0, stream>>>(
            d_pos_b[buf], d_perm, base, curB, (unsigned long long)n, d_oob);
        CHECK_CUDA_ERRORS;

        cur_buf = 1 - cur_buf;
    }

    cudaStreamSynchronize(stream);
    CHECK_CUDA_ERRORS;

    // Check invalid/oob counters
    unsigned int h_invalid = 0, h_oob = 0;
    cudaMemcpy(&h_invalid, d_invalid, sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_oob, d_oob, sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERRORS;
    if (h_invalid != 0 || h_oob != 0) {
        fprintf(stderr, "[reorder] WARNING: invalid_assign=%u, oob_scatter=%u\n",
                h_invalid, h_oob);
    }

    // Copy perm back if requested
    if (h_perm_out) {
        cudaMemcpy(h_perm_out, d_perm, sizeof(int) * (size_t)n,
                   cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERRORS;
    }

    // Cleanup
    for (int t = 0; t < 2; ++t) {
        cudaFree(d_pos_b[t]);
    }
    cudaFree(d_counts);
    cudaFree(d_writeptr);
    cudaFree(d_perm);
    cudaFree(d_invalid);
    cudaFree(d_oob);
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
    float* h_objective,            // optional output
    const int* h_indices_in,       // [n] CPU 原始索引数组（可选，如果为nullptr则跳过索引重排）
    int* h_indices_out             // [n] CPU 重排后的索引数组（可选，如果为nullptr则跳过索引重排）
) {
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;
    
    // 参数验证
    if (!h_data_in || !h_data_out || !d_centroids || n <= 0 || dim <= 0 || k <= 0) {
        fprintf(stderr, "[ivf_kmeans] ERROR: Invalid parameters\n");
        return false;
    }
    
    cudaSetDevice(device_id);
    CHECK_CUDA_ERRORS;
    
    // Step 1: Allocate device buffer for assignments
    int* d_assign = nullptr;
    cudaMalloc(&d_assign, sizeof(int) * (size_t)n);
    CHECK_CUDA_ERRORS;
    
    // Step 2: Run K-means clustering
    float obj = 0.0f;
    if (use_minibatch) {
        gpu_kmeans_minibatch(cfg, h_data_in, d_assign, d_centroids, &obj);
    } else {
        gpu_kmeans_lloyd(cfg, h_data_in, d_assign, d_centroids, &obj);
    }
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    
    if (h_objective) {
        *h_objective = obj;
    }
    
    // Step 3: Build permutation on GPU (直接使用设备指针，避免 H2D 传输)
    std::vector<int> h_perm(n);
    ClusterInfo cluster_info;
    ClusterInfo* info_ptr = h_cluster_info ? h_cluster_info : &cluster_info;
    
    // 使用设备指针版本，直接传入 d_assign，避免中间的 H2D 拷贝
    gpu_build_permutation_by_cluster(cfg, d_assign, h_perm.data(), info_ptr,
                                    device_id, batch_size, 0);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS;
    
    // Step 5: Reorder vectors on CPU
    cpu_reorder_vectors_by_permutation(h_data_in, h_perm.data(), h_data_out, n, dim);
    
    // Step 6: Reorder indices on CPU (if provided)
    if (h_indices_in && h_indices_out) {
        cpu_reorder_indices_by_permutation(h_indices_in, h_perm.data(), h_indices_out, n);
    }
    
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
