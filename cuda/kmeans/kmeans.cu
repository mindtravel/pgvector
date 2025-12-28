#include "kmeans.cuh"
#include "../utils.cuh"
#include "../l2norm/l2norm.cuh"
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ============================================================
// GPU Kernels Implementation
// ============================================================



__global__ void kernel_update_centroids(
    float* __restrict__ centroids,   // [k, dim]
    const float* __restrict__ accum, // [k, dim]
    const int* __restrict__ counts,  // [k]
    int k, int dim
) {
    int c = blockIdx.x;
    int j = threadIdx.x;
    if (c >= k) return;

    int cnt = counts[c];
    if (cnt <= 0) return;  // keep old centroid
    float inv = 1.0f / (float)cnt;

    for (int col = j; col < dim; col += blockDim.x) {
        centroids[(size_t)c * dim + col] = accum[(size_t)c * dim + col] * inv;
    }
}

// ============================================================
// GEMM-based KMeans Kernels
// ============================================================

/**
 * Kernel: 初始化最佳匹配（best_dist2 = INF, best_idx = 0）
 */
__global__ void kernel_init_best(
    float* __restrict__ best_dist2,
    int* __restrict__ best_idx,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        best_dist2[i] = 3.402823e38f;  // FLT_MAX
        best_idx[i] = 0;
    }
}

/**
 * Kernel: 从 GEMM 结果（col-major dotT）更新最佳匹配（优化版本：使用规约）
 * 
 * dotT: [curK, curB] col-major，即 dotT[t + i*curK] = dot(centroid[cbase+t], data[i])
 * 
 * 优化策略：
 * - 每个线程处理一个数据点
 * - 使用 warp shuffle 在 warp 内进行 min 规约
 * - 使用循环展开和向量化加载优化内存访问
 */
__global__ void kernel_update_best_from_dotT(
    const float* __restrict__ dotT,      // [curK, curB] col-major
    const float* __restrict__ xnorm2,     // [curB]
    const float* __restrict__ cnorm2_global,  // [k] 全局centroid范数
    int curB,
    int curK,
    int cbase,                            // centroid起始偏移
    int* __restrict__ best_idx,          // [curB]
    float* __restrict__ best_dist2        // [curB]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= curB) return;

    const float xn = xnorm2[i];
    float bestd = best_dist2[i];
    int bestc = best_idx[i];

    // 使用循环展开优化：每次处理多个centroid
    const int unroll_factor = 4;
    int t = 0;
    
    // 主循环：处理大部分centroid
    for (; t + unroll_factor <= curK; t += unroll_factor) {
        #pragma unroll
        for (int u = 0; u < unroll_factor; ++u) {
            int tidx = t + u;
            float dot = dotT[tidx + (size_t)i * curK];
            float cn = cnorm2_global[cbase + tidx];
            float d2 = xn + cn - 2.f * dot;
            int cid = cbase + tidx;
            if (d2 < bestd) {
                bestd = d2;
                bestc = cid;
            }
        }
    }
    
    // 处理剩余的centroid
    for (; t < curK; ++t) {
        float dot = dotT[t + (size_t)i * curK];
        float cn = cnorm2_global[cbase + t];
        float d2 = xn + cn - 2.f * dot;
        int cid = cbase + t;
        if (d2 < bestd) {
            bestd = d2;
            bestc = cid;
        }
    }

    best_dist2[i] = bestd;
    best_idx[i] = bestc;
}

/**
 * Kernel: 从 GEMM 结果（col-major dotT）更新最佳匹配（Warp规约优化版本）
 * 
 * 使用 warp shuffle 进行规约，适用于 curK 较大的情况
 * 每个 warp 协作处理一个数据点，在 warp 内进行 min 规约
 */
__global__ void kernel_update_best_from_dotT_warp_reduce(
    const float* __restrict__ dotT,      // [curK, curB] col-major
    const float* __restrict__ xnorm2,     // [curB]
    const float* __restrict__ cnorm2_global,  // [k] 全局centroid范数
    int curB,
    int curK,
    int cbase,                            // centroid起始偏移
    int* __restrict__ best_idx,          // [curB]
    float* __restrict__ best_dist2        // [curB]
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_count = (blockDim.x + 31) / 32;
    const int data_idx = blockIdx.x * warp_count + warp_id;
    
    if (data_idx >= curB) return;

    const float xn = xnorm2[data_idx];
    float bestd = (lane_id == 0) ? best_dist2[data_idx] : 3.402823e38f;  // FLT_MAX
    int bestc = (lane_id == 0) ? best_idx[data_idx] : -1;

    // 每个线程处理 curK / 32 个centroid（向上取整）
    const int centroids_per_thread = (curK + 31) / 32;
    const int start_t = lane_id * centroids_per_thread;
    const int end_t = (start_t + centroids_per_thread < curK) ? (start_t + centroids_per_thread) : curK;

    // 每个线程处理分配给它的centroid
    for (int t = start_t; t < end_t; ++t) {
        float dot = dotT[t + (size_t)data_idx * curK];
        float cn = cnorm2_global[cbase + t];
        float d2 = xn + cn - 2.f * dot;
        int cid = cbase + t;
        if (d2 < bestd) {
            bestd = d2;
            bestc = cid;
        }
    }

    // Warp shuffle 规约：找到 warp 内的最小值
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_d = __shfl_down_sync(0xffffffff, bestd, offset);
        int other_c = __shfl_down_sync(0xffffffff, bestc, offset);
        if (other_d < bestd) {
            bestd = other_d;
            bestc = other_c;
        }
    }

    // Lane 0 写入结果
    if (lane_id == 0) {
        best_dist2[data_idx] = bestd;
        best_idx[data_idx] = bestc;
    }
}

/**
 * Kernel: 从分配结果累加（用于 GEMM 版本）
 */
__global__ void kernel_accum_from_assign(
    const float* __restrict__ data,   // [n, dim]
    int n, int dim,
    const int* __restrict__ assign, // [n]
    float* __restrict__ accum,    // [k, dim]
    int* __restrict__ counts      // [k]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = assign[i];
    atomicAdd(&counts[c], 1);
    float* acc = accum + (size_t)c * dim;
    int base = i * dim;
    for (int j = 0; j < dim; ++j) {
        atomicAdd(&acc[j], data[base + j]);
    }
}

static inline void cublas_check(cublasStatus_t st, const char* msg) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublas error %d at %s\n", (int)st, msg);
        std::abort();
    }
}

// ============================================================
// GPU KMeans Runner Implementation (GEMM-optimized)
// ============================================================

void gpu_kmeans_lloyd(
    const KMeansCase& cfg,
    const float* d_data,             // [n, dim] row-major
    int* d_assign,                   // [n]
    float* d_centroids,              // [k, dim] (in/out)
    float* h_objective                // sum dist2
) {
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;

    // ====== 可调参数：batch & ktile ======
    const int Ktile = 8192;  // 每次GEMM处理的centroid数量
    const int B = 1 << 20;   // 每次处理的点数量（1M）

    // ====== buffer ======
    float* d_accum = nullptr;      // [k, dim]
    int*   d_counts = nullptr;     // [k]
    float* d_cnorm2 = nullptr;     // [k]

    // 每个batch：xnorm2[B], best_dist2[B], best_idx[B], dot[B*Ktile]
    float* d_xnorm2 = nullptr;
    float* d_best_dist2 = nullptr;
    int*   d_best_idx = nullptr;
    float* d_dot = nullptr;        // [B, Ktile] 但实际存的是col-major [curK, curB]

    // d_centroids 是输入参数，不需要在这里分配
    cudaMalloc(&d_accum,     sizeof(float) * (size_t)k * dim);
    cudaMalloc(&d_counts,    sizeof(int)   * (size_t)k);
    cudaMalloc(&d_cnorm2,    sizeof(float) * (size_t)k);

    cudaMalloc(&d_xnorm2,    sizeof(float) * (size_t)B);
    cudaMalloc(&d_best_dist2,sizeof(float) * (size_t)B);
    cudaMalloc(&d_best_idx,  sizeof(int)   * (size_t)B);
    cudaMalloc(&d_dot,       sizeof(float) * (size_t)B * (size_t)Ktile);


    // ====== cuBLAS handle ======
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle), "cublasCreate");
    cublas_check(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH), "setMathMode");

    // ====== 常量 ======
    const float alpha = 1.f;
    const float beta  = 0.f;

    // ====== iteration ======
    for (int it = 0; it < cfg.iters; ++it) {
        // clear accum & counts
        dim3 fill_block(256);
        int fill_grid_accum = std::min(65535, (int)(((size_t)k * dim + fill_block.x - 1) / fill_block.x));
        int fill_grid_counts = std::min(65535, (int)((k + fill_block.x - 1) / fill_block.x));
        fill_kernel<<<fill_grid_accum, fill_block>>>(d_accum, 0.0f, (int)((size_t)k * dim));
        fill_int_kernel<<<fill_grid_counts, fill_block>>>(d_counts, 0, k);

        compute_l2_squared_gpu(d_centroids, d_cnorm2, k, dim, L2NORM_AUTO);

        // ====== 分块处理数据点 ======
        double obj_sum = 0.0;

        for (int base = 0; base < n; base += B) {
            int curB = std::min(B, n - base);

            // xnorm2 for this batch
            compute_l2_squared_gpu(d_data + (size_t)base * dim, d_xnorm2, curB, dim, L2NORM_AUTO);

            // init best_dist2 = +inf, best_idx = 0
            {
                int threads = 256;
                int blocks = (curB + threads - 1) / threads;
                kernel_init_best<<<blocks, threads>>>(d_best_dist2, d_best_idx, curB);
            }

            // ====== centroid 分块 Ktile ======
            for (int cbase = 0; cbase < k; cbase += Ktile) {
                int curK = std::min(Ktile, k - cbase);

                const float* A = d_centroids + (size_t)cbase * dim;
                const float* Bm = d_data + (size_t)base * dim;
                float* Cc = d_dot;

                // Treat row-major C[curK,dim] as col-major Ccm[dim,curK]
                // Treat row-major X[curB,dim] as col-major Xcm[dim,curB]
                // Compute dotT[curK,curB] (col-major) = Ccm^T[curK,dim] * Xcm[dim,curB]
                cublas_check(
                    cublasSgemm(
                        handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,     // A^T * B
                        curK, curB, dim,              // m=curK, n=curB, k=dim
                        &alpha,
                        A, dim,                       // A is Ccm with shape [dim,curK], lda=dim
                        Bm, dim,                      // B is Xcm with shape [dim,curB], ldb=dim
                        &beta,
                        Cc, curK                     // C is [curK,curB] col-major, ldc=curK
                    ),
                    "cublasSgemm(Ccm^T * Xcm)"
                );


                // 更新best
                {
                    int threads = 256;
                    int blocks = (curB + threads - 1) / threads;
                    kernel_update_best_from_dotT<<<blocks, threads>>>(
                        d_dot, d_xnorm2, d_cnorm2,
                        curB, curK, cbase,
                        d_best_idx, d_best_dist2
                    );
                }
            }

            // 写回assign（全局）
            cudaMemcpy(d_assign + base, d_best_idx, sizeof(int) * (size_t)curB, cudaMemcpyDeviceToDevice);

            // accum + counts
            {
                int threads = 256;
                int blocks = (curB + threads - 1) / threads;
                kernel_accum_from_assign<<<blocks, threads>>>(
                    d_data + (size_t)base * dim, curB, dim,
                    d_best_idx, d_accum, d_counts
                );
            }

            // objective
            if (h_objective) {
                std::vector<float> h_tmp(curB);
                cudaMemcpy(h_tmp.data(), d_best_dist2, sizeof(float) * (size_t)curB, cudaMemcpyDeviceToHost);
                for (int i = 0; i < curB; ++i) obj_sum += (double)h_tmp[i];
            }
        }

        // update centroids: one block per centroid
        kernel_update_centroids<<<k, 256>>>(d_centroids, d_accum, d_counts, k, dim);

        if (h_objective) *h_objective = (float)obj_sum;
    }


    cublasDestroy(handle);

    // d_centroids 是输入参数，不需要在这里释放
    cudaFree(d_accum);
    cudaFree(d_counts);
    cudaFree(d_cnorm2);
    cudaFree(d_xnorm2);
    cudaFree(d_best_dist2);
    cudaFree(d_best_idx);
    cudaFree(d_dot);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in gpu_kmeans_lloyd: %s\n", cudaGetErrorString(err));
    }
}


// ============================================================
// CPU Initialization Functions Implementation
// ============================================================

__host__ void init_centroids_by_sampling(
    const KMeansCase& cfg,
    const float* data,        // [n, dim]
    float* out_centroids      // [k, dim]
) {
    std::mt19937 rng(cfg.seed);
    
    // 生成索引序列 [0, 1, 2, ..., n-1]
    std::vector<int> indices(cfg.n);
    for (int i = 0; i < cfg.n; ++i) {
        indices[i] = i;
    }
    
    // Fisher-Yates 洗牌算法：随机打乱前 k 个位置
    // 这样前 k 个索引就是随机选择的 k 个不同的点
    for (int i = 0; i < cfg.k && i < cfg.n; ++i) {
        std::uniform_int_distribution<int> dist(i, cfg.n - 1);
        int j = dist(rng);
        std::swap(indices[i], indices[j]);
    }
    
    // 复制前 k 个点作为聚类中心
    for (int c = 0; c < cfg.k; ++c) {
        int idx = indices[c];
        std::memcpy(out_centroids + (size_t)c * cfg.dim,
                    data + (size_t)idx * cfg.dim,
                    sizeof(float) * cfg.dim);
    }
}

