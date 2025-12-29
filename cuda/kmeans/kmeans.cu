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
#include <../unit_tests/common/test_utils.cuh>

// ============================================================
// GPU Kernels Implementation
// ============================================================

/**
 * Kernel: 对 best_dist2 数组求和（block reduce）
 * 使用共享内存进行 block 内规约，然后原子加到一个全局累加器
 */
__global__ void kernel_reduce_sum(
    const float* __restrict__ data,
    float* __restrict__ output,  // 单个 float 的累加器
    int n
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程加载一个元素
    float val = (i < n) ? data[i] : 0.0f;
    
    // Block 内规约
    sdata[tid] = val;
    __syncthreads();
    
    // 规约：使用 warp shuffle + shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Block 0 的第一个线程原子加
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

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
    const float* h_data,             // [n, dim] row-major，主机端数据
    int* d_assign,                   // [n]
    float* d_centroids,              // [k, dim] (in/out)，常驻显存
    float* h_objective                // sum dist2
) {
    const int n = cfg.n, dim = cfg.dim, k = cfg.k;

    // ====== 可调参数：batch & ktile ======
    // 减小 B 和 Ktile 以控制 d_dot 显存占用
    // 目标：d_dot = B * Ktile <= 1GB (约 256M floats)
    const int Ktile = 4096;  // 每次GEMM处理的centroid数量（从8192减小到4096）
    const int B = 1 << 15;   // 每次处理的点数量（32K，从256K减小到32K）
    // d_dot = 32K * 4K = 128M floats ≈ 512MB（可接受）

    // ====== buffer ======
    float* d_accum = nullptr;      // [k, dim]
    int*   d_counts = nullptr;     // [k]
    float* d_cnorm2 = nullptr;     // [k]

    // 每个batch：xnorm2[B], best_dist2[B], best_idx[B], dot[B*Ktile]
    float* d_xnorm2 = nullptr;
    float* d_best_dist2 = nullptr;
    int*   d_best_idx = nullptr;
    float* d_dot = nullptr;        // [B, Ktile] 但实际存的是col-major [curK, curB]

    // 双缓冲：两个GPU缓冲区交替使用
    float* d_data_buf[2] = {nullptr, nullptr};  // 双缓冲数据缓冲区
    cudaStream_t stream[2];                      // 两个CUDA流用于异步传输
    cudaEvent_t event[2];                        // 事件用于依赖管理，替代同步

    // d_centroids 是输入参数，不需要在这里分配
    cudaMalloc(&d_accum,     sizeof(float) * (size_t)k * dim);
    cudaMalloc(&d_counts,    sizeof(int)   * (size_t)k);
    cudaMalloc(&d_cnorm2,    sizeof(float) * (size_t)k);

    cudaMalloc(&d_xnorm2,    sizeof(float) * (size_t)B);
    cudaMalloc(&d_best_dist2,sizeof(float) * (size_t)B);
    cudaMalloc(&d_best_idx,  sizeof(int)   * (size_t)B);
    cudaMalloc(&d_dot,       sizeof(float) * (size_t)B * (size_t)Ktile);

    // 分配双缓冲
    cudaMalloc(&d_data_buf[0], sizeof(float) * (size_t)B * dim);
    cudaMalloc(&d_data_buf[1], sizeof(float) * (size_t)B * dim);
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    cudaEventCreate(&event[0]);
    cudaEventCreate(&event[1]);
    
    // GPU 上的 objective 累加器（单个 float）
    float* d_objective_sum = nullptr;
    if (h_objective) {
        cudaMalloc(&d_objective_sum, sizeof(float));
    }


    // ====== cuBLAS handle ======
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle), "cublasCreate");
    cublas_check(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH), "setMathMode");

    // ====== 常量 ======
    const float alpha = 1.f;
    const float beta  = 0.f;

    // 当前使用的缓冲区索引（0或1）
    int cur_buf = 0;

    // ====== iteration ======
    for (int it = 0; it < cfg.iters; ++it) {
        // clear accum & counts
        // dim3 fill_block(256);
        // int fill_grid_accum = std::min(65535, (int)(((size_t)k * dim + fill_block.x - 1) / fill_block.x));
        // int fill_grid_counts = std::min(65535, (int)((k + fill_block.x - 1) / fill_block.x));
        // fill_kernel<<<fill_grid_accum, fill_block>>>(d_accum, 0.0f, (int)((size_t)k * dim));
        // fill_int_kernel<<<fill_grid_counts, fill_block>>>(d_counts, 0, k);
        // 使用 stream[0] 来执行迭代开始的操作（accum/counts清零和centroid范数计算）
        // 这些操作需要在所有batch开始前完成
        cudaMemsetAsync(d_accum, 0, sizeof(float) * (size_t)k * dim, stream[0]);
        cudaMemsetAsync(d_counts, 0, sizeof(int) * (size_t)k, stream[0]);
        // 绑定到 stream[0]，确保在所有batch计算前完成
        compute_l2_squared_gpu(d_centroids, d_cnorm2, k, dim, L2NORM_AUTO, stream[0]);

        // ====== 分块处理数据点（使用双缓冲） ======
        cur_buf = 0;
        
        // 清零 objective 累加器（如果启用，使用异步版本）
        if (h_objective && d_objective_sum) {
            cudaMemsetAsync(d_objective_sum, 0, sizeof(float), stream[0]);
        }
        
        // 记录 stream[0] 上的初始化操作完成事件（accum清零、centroid范数计算）
        // 确保后续batch计算时 d_cnorm2 和 d_accum 已准备好
        cudaEvent_t init_event;
        cudaEventCreate(&init_event);
        cudaEventRecord(init_event, stream[0]);

        // 预加载第一个batch
        int first_base = 0;
        int first_curB = std::min(B, n - first_base);
        // 等待初始化操作完成，确保 d_cnorm2 和 d_accum 已准备好
        cudaStreamWaitEvent(stream[cur_buf], init_event, 0);
        cudaMemcpyAsync(d_data_buf[cur_buf], h_data + (size_t)first_base * dim,
                       sizeof(float) * (size_t)first_curB * dim,
                       cudaMemcpyHostToDevice, stream[cur_buf]);
        // 记录事件，用于后续依赖
        cudaEventRecord(event[cur_buf], stream[cur_buf]);

        for (int base = 0; base < n; base += B) {
            int curB = std::min(B, n - base);
            float* d_data_cur = d_data_buf[cur_buf];
            
            // 等待当前batch的数据传输完成（使用事件依赖）
            // 第一次迭代：等待预加载完成；后续迭代：等待上一个batch的计算完成
            cudaStreamWaitEvent(stream[cur_buf], event[cur_buf], 0);

            // 异步预加载下一个batch（如果还有）
            // 注意：这里先启动预加载，但会在计算完成后才真正传输（通过事件依赖）
            int next_base = base + B;
            int next_buf = 1 - cur_buf;

            // xnorm2 for this batch（在当前流的流上执行）
            compute_l2_squared_gpu(d_data_cur, d_xnorm2, curB, dim, L2NORM_AUTO, stream[cur_buf]);

            // init best_dist2 = +inf, best_idx = 0（在当前流的流上执行）
            {
                int threads = 256;
                int blocks = (curB + threads - 1) / threads;
                kernel_init_best<<<blocks, threads, 0, stream[cur_buf]>>>(d_best_dist2, d_best_idx, curB);
            }

            // ====== centroid 分块 Ktile ======
            for (int cbase = 0; cbase < k; cbase += Ktile) {
                int curK = std::min(Ktile, k - cbase);

                const float* A = d_centroids + (size_t)cbase * dim;
                const float* Bm = d_data_cur;
                float* Cc = d_dot;

                // 设置cuBLAS流
                cublas_check(cublasSetStream(handle, stream[cur_buf]), "cublasSetStream");

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


                // 更新best（在当前流的流上执行）
                {
                    int threads = 256;
                    int blocks = (curB + threads - 1) / threads;
                    kernel_update_best_from_dotT<<<blocks, threads, 0, stream[cur_buf]>>>(
                        d_dot, d_xnorm2, d_cnorm2,
                        curB, curK, cbase,
                        d_best_idx, d_best_dist2
                    );
                }
            }

            // 写回assign（全局，使用异步拷贝）
            cudaMemcpyAsync(d_assign + base, d_best_idx, sizeof(int) * (size_t)curB, 
                           cudaMemcpyDeviceToDevice, stream[cur_buf]);

            // accum + counts（在当前流的流上执行）
            {
                int threads = 256;
                int blocks = (curB + threads - 1) / threads;
                kernel_accum_from_assign<<<blocks, threads, 0, stream[cur_buf]>>>(
                    d_data_cur, curB, dim,
                    d_best_idx, d_accum, d_counts
                );
            }

            // objective：在 GPU 上做 reduce，避免回传到 host
            if (h_objective && d_objective_sum) {
                int reduce_threads = 256;
                int reduce_blocks = (curB + reduce_threads - 1) / reduce_threads;
                int reduce_shmem = reduce_threads * sizeof(float);
                kernel_reduce_sum<<<reduce_blocks, reduce_threads, reduce_shmem, stream[cur_buf]>>>(
                    d_best_dist2, d_objective_sum, curB);
            }

            // 记录当前batch计算完成事件（用于下一个batch的依赖）
            cudaEventRecord(event[cur_buf], stream[cur_buf]);
            
            // 现在可以安全地预加载下一个batch（等待当前batch计算完成）
            if (next_base < n) {
                int next_curB = std::min(B, n - next_base);
                // 等待当前batch的计算完成，避免覆盖正在使用的缓冲区
                cudaStreamWaitEvent(stream[next_buf], event[cur_buf], 0);
                cudaMemcpyAsync(d_data_buf[next_buf], h_data + (size_t)next_base * dim,
                               sizeof(float) * (size_t)next_curB * dim,
                               cudaMemcpyHostToDevice, stream[next_buf]);
                // 记录传输完成事件（用于下一个迭代的依赖）
                cudaEventRecord(event[next_buf], stream[next_buf]);
            }
            
            // 切换到下一个缓冲区
            cur_buf = next_buf;
        }

        // 等待所有batch完成（只同步必要的流）
        // 等待两个流都完成，确保所有batch的计算和accum都完成
        cudaStreamSynchronize(stream[0]);
        cudaStreamSynchronize(stream[1]);
        
        // update centroids: one block per centroid（使用 stream[0]）
        kernel_update_centroids<<<k, 256, 0, stream[0]>>>(d_centroids, d_accum, d_counts, k, dim);
        
        // 等待 centroid 更新完成（只同步 stream[0]）
        cudaStreamSynchronize(stream[0]);

        // 从 GPU 读取 objective 值（只在迭代结束时读取一次）
        if (h_objective && d_objective_sum) {
            float obj_val = 0.0f;
            cudaMemcpy(&obj_val, d_objective_sum, sizeof(float), cudaMemcpyDeviceToHost);
            *h_objective = obj_val;
            COUT_ENDL("iter=", it, "obj=", obj_val);
        }
        
        // 清理临时事件
        cudaEventDestroy(init_event);
    }


    cublasDestroy(handle);

    // 释放双缓冲
    cudaFree(d_data_buf[0]);
    cudaFree(d_data_buf[1]);
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaEventDestroy(event[0]);
    cudaEventDestroy(event[1]);
    
    if (d_objective_sum) {
        cudaFree(d_objective_sum);
    }

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

